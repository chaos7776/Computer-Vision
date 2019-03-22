# Computer-Vision
计算机视觉代码

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
  
  
[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)

### 空洞卷积网络



