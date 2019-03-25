# Computer-Vision
计算机视觉代码


## 图片分类

## 物体识别

## 语义分割

此部分内容主要介绍卷积神经网络在语义分割中的引用

---
数据集：

  - [VOC2012](http://cocodataset.org/#home)    
  - [MSCOCO](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
  
  
### 全连接卷积网络

2014年，加州大学伯克利分校的Long等人在[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)提出的完全卷积网络，推广了原有的CNN结构，在不带有全连接层的情况下能进行密集预测。

  - 将端到端的卷积网络推广到语义分割中；
  - 重新将预训练好的Imagenet网络用于分割问题中；
  - 使用反卷积层进行上采样；
  - 提出了跳跃连接来改善上采样的粗糙程度。
  
1x1卷积最早出现在Network In Network中，代替模型最后面的全连接层，主要有两方面的好处：
- 改变维度
全连接层使得数据扁平化，丢失了图片的空间信息，而1X1卷积可以改变维度（可以增加维度和降低维度），保留了空间信息
- 减少模型的参数
由于卷积网络可以共享参数，模型参数变少。以GoogLenet中Inception模型计算如下：


下图是其[论文](https://arxiv.org/abs/1409.4842)对GoogLeNet的可视化

  
  
[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)

### 空洞卷积网络



