from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy

import torch.nn.functional as F


from torch.autograd import Variable


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, optimizer_t, opt):
    """One epoch distillation"""
    for module in module_list:
        module.train()
        module_list[2].eval() # model_t eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    stud_net = module_list[1]
    model_t = module_list[2]
    teac_net = module_list[3]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_t = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_t = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            clip_img, input, target, index, contrast_idx = data
        else:
            clip_img, input, target, index = data # clip_img 224, input 32
        data_time.update(time.time() - end)        
        clip_img = clip_img.float()
        input = input.float()
        
        if torch.cuda.is_available():
            clip_img = clip_img.cuda()
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s = model_s(input, is_feat=True, preact=False)
        with torch.no_grad():
            feat_t = model_t.encode_image(clip_img).float()
            feat_t = feat_t.detach()

        # logits for classification 
        fs, logit_s = stud_net(feat_s)
        ft, logit_t = teac_net(feat_t)

        loss_cls = criterion_cls(logit_s, target)
        loss_cls_t = criterion_cls(logit_t, target)

        loss_div = criterion_div(logit_s, logit_t)
        
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'ega':
            loss_kd = criterion_kd(fs, ft)
        elif opt.distill == 'rkd':            
            loss_kd = criterion_kd(fs, ft)
        elif opt.distill == 'pkt':            
            loss_kd = criterion_kd(fs, ft)
        elif opt.distill == 'nce':
            loss_kd = criterion_kd(fs, ft, target)
        elif opt.distill == 'ccl':
            loss_kd = criterion_kd(fs, ft, target, logit_s, logit_t)        
        elif opt.distill == 'crd':
            loss_kd = criterion_kd(fs, ft, index, contrast_idx)
        elif opt.distill == 'hint':
            loss_kd = criterion_kd(fs, ft) 
        
        loss = opt.gamma * loss_cls + opt.beta * loss_kd + opt.alpha * loss_div
        loss_t = loss_cls_t

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        
        # for teacher
        acc1_t, _ = accuracy(logit_t, target, topk=(1, 5))
        losses_t.update(loss_t.item(), input.size(0))
        top1_t.update(acc1_t[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer_t.zero_grad()
        loss_t.backward()

        optimizer_t.step()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('training student info')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Train Acc@1 {top1.avg:.3f} Loss_avg {loss.avg:.4f}'
          .format(top1=top1, loss=losses))
 
    return top1.avg, losses.avg


def validate(val_loader, module_list, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_t = AverageMeter()
    top1_t = AverageMeter()

    # switch to evaluate mode
    for module in module_list:
        module.eval()
    model_s = module_list[0]
    stud_net = module_list[1]
    model_t = module_list[2]
    teac_net = module_list[3]

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            if opt.distill in ['crd']:
                clip_img, input, target, index = data
            else:
                clip_img, input, target, index = data
            
            input = input.float()
            clip_img = clip_img.float()

            if torch.cuda.is_available():
                clip_img = clip_img.cuda()
                input = input.cuda()
                target = target.cuda()

            # compute output
            feat_s = model_s(input)
            fs, output = stud_net(feat_s)
            loss = criterion(output, target)

            #compute teacher model
            feat_t = model_t.encode_image(clip_img).float()
            feat_t = feat_t.detach()
            ft, output_t = teac_net(feat_t)
            loss_t = criterion(output_t, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            #for teacher
            acc1_t, _ = accuracy(output_t, target, topk=(1, 5))
            losses_t.update(loss_t.item(), input.size(0))
            top1_t.update(acc1_t[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Val Acc@1 {top1.avg:.3f} Loss_avg {loss.avg:.4f}'
          .format(top1=top1, loss=losses))

    return top1.avg, top5.avg, losses.avg
