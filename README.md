# MXNet/Gluon for CIFAR-10 Dataset
## Introduction

This repository is about resnet164 and densenet architecture's gluon implement for cifar10 dataset in kaggle.

![](https://ws4.sinaimg.cn/large/006tKfTcly1fl9tro18vxj30m80ai0ue.jpg)

I just use mxnet/gluon to implement all of these models, and the single model is near rank1 in leader board, ensemble model is over rank1. All those ideas come from [gluon community](https://discuss.gluon.ai/t/topic/1545/451), welcome to join big family of gluon.

![](https://ws2.sinaimg.cn/large/006tKfTcly1fl9sytts8ej30qm05gwf0.jpg)



## Requirements

- [MXNet(0.12)](https://mxnet.incubator.apache.org/versions/master/) 

  fast, flexible and portable deep learning architecture

- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) 

  for visualization loss and accuracy

### Architectures and papers

- Resnet164
  - [Identity Mappings in Deep Residual Networks](https://pdfs.semanticscholar.org/912e/bc25b50775e726b18dcc768acfe553158b83.pdf)
- Densenet
  - [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)



## Accuracy of single model

Before training, we will do standard data augumentation, pad 4 and random crop to 32 image size, do random mirror transform. 

### Resnet164

This model is defined in [resnet.py](https://github.com/SherlockLiao/cifar10-gluon/blob/master/resnet.py), training file is train_resnet164.ipynb. The training strategy is same as the paper, total epochs are 200, batch size is 128, initial learing rate is 0.1, momentum is 0.9, learning rate decay at 90 epoch and 140 epoch.

![](https://ws3.sinaimg.cn/large/006tKfTcly1fl9t9cljgcj30jk070gm6.jpg)

After 200 epochs, training accuracy is almost 100%, kaggle score is 0.9526.

### Densnet

This model is defined in [densenet.py](https://github.com/SherlockLiao/cifar10-gluon/blob/master/densenet.py), training file is train_densenet.ipynb. The training strategy is similary as resnet, total epochs are 300, batch size is 128, initial learning rate is 0.1, momentum is 0.9, learning rate decay at 50% and 75% of total epochs.

![](https://ws3.sinaimg.cn/large/006tKfTcly1fl9t9e7a3pj30jb05rwez.jpg)

After 300 epochs, training accuracy is almost 100%, kaggle score is 0.9536.



### Ensemble

We can ensemble these two models, get the result of each model, then compute the final result by weighted each model's output. The weight is the accuracy of each single model. The ensemble result is in ensemble_submission.ipynb, the final result is 0.9616.



## Future work

- Use more models and data augumentation to do ensemble can get a better result.


- There is a paper about [mixup](https://arxiv.org/pdf/1710.09412.pdf) show that it can over 97% at cifar10 dataset, so if I have time, I want to try this strategy. As in paper, this strategy is like a way of data augumentation.