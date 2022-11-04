"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, RKDLoss, PKT
from crd.criterion import CRDLoss
from distiller_zoo import NCELoss, EGA, CCL

from helper.loops import train_distill as train, validate
from helper.pretrain import init

import clip

from models.stud_net import StudNet
from models.teac_net import TeacNet

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='student net learning rate')
    parser.add_argument('--learning_rate_t', type=float, default=0.01, help='teacher layers learning rate')
    
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','stl10', 'tinyimagenet'], help='dataset')
    parser.add_argument('--n_cls', type=int, default=100, help='number of the class for dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8x4',
                        choices=['resnet8x4', 'resnet32x4', 'vgg13', 'ResNet50','ShuffleV1'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # Add other teacher networks
    parser.add_argument('--clip_tn', type=int, default=1, help='use clip teacher network')
    parser.add_argument('--clip_mode', default='ViT-B/32', type=str, choices=['RN50', 'RN101','ViT-B/32','ViT-B/16'])
    #parser.add_argument('--image_size', type=int, default=224, help='image size can be reshaped for pretrained models')
    parser.add_argument('--stud_dim', type=int, default=256, help='the raw student feature dim before linear layer')
    parser.add_argument('--feature_dim', type=int, default=256, help='the space dimension which fs and ft joint map to')
 
    # distillation
    parser.add_argument('--distill', type=str, default='nce', choices=['kd', 'hint', 'crd', 'rkd', 'pkt', 
                                                                      'nce', 'ccl', 'ega'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')    
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=1, help='weight balance for other distillation losses')

    parser.add_argument('-node', '--node_weight', type=float, default=0.8, help='weight for node matching loss')
    parser.add_argument('-edge', '--edge_weight', type=float, default=0.24, help='weight for edge matching loss')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['ShuffleV1']:
        opt.learning_rate = 0.01

    # set the path according to the environment   
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.clip_tn: # use pretrained_clip
        if opt.clip_mode == 'ViT-B/32': # for the convenience of making dir.
            opt.model_t = 'vit32'
        elif opt.clip_mode == 'ViT-B/16':
            opt.model_t = 'vit16'
        else:
            opt.model_t = opt.clip_mode # RN101 or RN50.      
    else:
        opt.model_t = get_teacher_name(opt.path_t)
        
    opt.model_name = 'S:{}_T:{}_f:{}_{}_{}_r:{}_b:{}'.format(opt.model_s, opt.model_t, opt.feature_dim, opt.dataset, opt.distill,
                                                            opt.gamma, opt.beta)
    
    
    opt.model_name = 'S:{}_T:{}_f:{}_{}_{}_r:{}_b:{}_node:{}_edge:{}_batchsize:{}'.format(opt.model_s, opt.model_t, opt.feature_dim, opt.dataset, opt.distill,
                                                            opt.gamma, opt.beta, opt.node_weight, opt.edge_weight, opt.batch_size)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt
def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def main():
    best_acc = 0

    opt = parse_option()


    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        opt.n_cls = 100
        print('train_loader length', len(train_loader))
        print('val_loader length', len(val_loader))

    else:
        raise NotImplementedError(opt.dataset)

    # model
    # confirm student feature dimension before linear layer.
    if opt.model_s in ['resnet8x4']:
        opt.stud_dim = 256
    elif opt.model_s in ['vgg13']:
        opt.stud_dim = 512
    elif opt.model_s in ['ShuffleV1']:
        opt.stud_dim = 960
    else:
        print('stud_dim is default dim 256')

    #load teacher model
    if opt.clip_tn:
        print('Use clip teacher networks.')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_t, preprocess = clip.load(opt.clip_mode, device=device)
        print('pretrained clip model is:', opt.clip_mode)
        print('student network is:', opt.model_s)
        print('joint space dim both fs and ft map to is:', opt.feature_dim)

        #frozen clip pretrained model
        for param in model_t.parameters():
            param.requires_grad = False

        if opt.clip_mode in ['RN50']:
            teac_net = TeacNet(1024, opt.feature_dim)
            stud_net = StudNet(opt.stud_dim, opt.feature_dim) 

        if opt.clip_mode in ['RN101', 'ViT-B/32', 'ViT-B/16']:
            teac_net = TeacNet(512, opt.feature_dim) 
            stud_net = StudNet(opt.stud_dim, opt.feature_dim)
    else:
        
        teac_net.load_state_dict(torch.load(opt.path_t)['model'])
        print('load pretrained teacher mode', opt.path_t)


    # get student model.
    model_s = model_dict[opt.model_s](num_classes=opt.n_cls)
    
    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])

    module_list.append(model_s)
    trainable_list.append(model_s)
    
    module_list.append(stud_net) 
    trainable_list.append(stud_net)

    module_list.append(model_t) 
    module_list.append(teac_net) 
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    print('distill loss is:', opt.distill)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)

    elif opt.distill == 'ega':
        criterion_kd = EGA(opt.node_weight, opt.edge_weight)
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'nce':
        criterion_kd = NCELoss(temperature=0.1)
    elif opt.distill == 'ccl':
        criterion_kd = CCL()
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()        
    elif opt.distill == 'crd':
        opt.s_dim = opt.feature_dim
        opt.t_dim = opt.feature_dim
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # student optimizer:
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    #teacher optimizer:
    optimizer_t = optim.SGD(teac_net.parameters(),
                          lr=opt.learning_rate_t,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        module_list.cuda()
        trainable_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, optimizer_t, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        # test
        test_acc, tect_acc_top5, test_loss = validate(val_loader, module_list, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'layer': stud_net.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'layer': stud_net.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'layer': stud_net.state_dict()
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
