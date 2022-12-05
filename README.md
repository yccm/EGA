# EGA
PyTorch Implementation on Paper [BMVC2022] Distilling Knowledge from Self-Supervised Teacher by Embedding Graph Alignment [Paper Link](https://arxiv.org/abs/2211.13264)


## Introduction

In this work, to leverage the strength of self-supervised pre-trained networks, we explore
the knowledge distillation paradigm to transfer knowledge
from a self-supervised pre-trained teacher model to a small, lightweight supervised
student network. We propose to model the instance-instance relationships by a graph structure in the
embedding space and distill such the structural information among instances by aligning
the teacher graph and the student graph – named as Embedding Graph Alignment (EGA).


<img src="https://github.com/yccm/EGA/blob/main/figure/model.png" width=95% height=95%>


## Setup
### Installation:
`python 3.7.11 
pytorch 1.7.1
numpy 1.20.3`

### Getting started:

### Prerequisites:
Fetch the pretrained teacher models. For example, use [CLIP](https://github.com/openai/CLIP) model.
Dowload the model and save it to `./CLIP/`.

#### Training on Cifar100 (as example):
An example of running Embeeding Graph Alignment (EGA) distillation is given by `./scripts/run_vit32.sh`.
Default dataset is `cifar100`. Default teacher model is clip `ViT-B/32`. Default student model is `resnet8x4`.
Some arguments for training are explained below:

`--clip_mode`: specify the type of clip teacher model.

`--model_s`: specify the student model.

`--distill`: specify the distillation method

`-r`: the weight of the cross-entropy loss for classification, default: 1

`-a`: the weight of the KD loss, default: None

`-b`: the weight of other distillation losses, default: 1.

## Bibtex
@article{ma2022distilling,
  title={Distilling Knowledge from Self-Supervised Teacher by Embedding Graph Alignment},
  author={Ma, Yuchen and Chen, Yanbei and Akata, Zeynep},
  journal={arXiv preprint arXiv:2211.13264},
  year={2022}
}


## Acknowledgement
This repo is based on the implementation of [CRD](https://github.com/HobbitLong/RepDistiller).
