## 0

按照设想，尝试成为一个数据科学家。

## 1 技能表

暂定技能表，后续会细化，并补充所有技能对应的博客链接和对应代码。

|层级|说明|技能|
|---|---|---|
|9|商业分析|soft-skill|
|8|系统实现|Web系统服务|
|7|数据搜索,推荐|搜索引擎设计，搜索系统，SEO，推荐系统|
|6|数据生成|文本生成，语音合成，图像生成|
|5|可视化展现|数据表报，复杂网络数据关系展现|
|4|数据挖掘/算法|普通数据挖掘算法，自然语言处理，语音`识别`,`合成`算法，图像`分类`,`定位`,`生成`算法|
|3|数据分析处理|大数据处理，普通数据处理，文本数据处理，语音数据处理，图像处理|
|2|数据存储|SQL,YARN,HBase,Hive,MongBD|
|1|数据采集分发|kafka,ZeroMQ|
|0| 数据源|普通数据集，公开数据集，爬虫数据|
|-1|数学基本|概率论与数据统计，线性代数，基础高数|



## 0.0 数据源

### 0.1 普通数据集

即被交付给的原始数据集

### 0.2 公开数据集

+ [uci数据集](http://archive.ics.uci.edu/ml/datasets.html) :一个公开的用于机器学习的数据集，包含几乎所有类别数据，适用于多种机器学习任务
+ 图像数据
  - coco
  - voc
  - 待补充..
  - 医疗影像数据
+ 语音数据
   - 中文语音
   - 英文语音

+ 公开视频数据
+ 金融股票数据

### 0.3 爬虫数据

+ HTML语言简介
+ python BeautifulSoup简介
+ scrapy简介

## 1 数据采集分发
#### 1.1 kafka

+ kafka 简介
+ Java 调用kafka
+ kafka原理

### 1.2 ZeroMQ

+ ZeroMQ 简介
+ ZeroMQ使用

## 2 数据储存

### 2.1 SQL

+ SQL基础语法
+ SQL高级语法
+ SQL优化
+ SQL思考

### 2.2 YARN

+ YARN生态简介

### 2.3 HDFS

+ HDFS文件系统简介

### 2.4 HBase

+ HBase 简介和安装
+ HBase java API简单调用
+ HBase 索引机制，二级索引设计
+ HBase 设计原理

### 2.5 Hive

+ Hive简介和安装
+ Hive语法，存储
+ Hive设计

### 2.6 MongoDB

+ MongoDB 简介和安装
+ MongoDB 设计

## 3 数据分析

### 3.1 普通数据处理
### 3.2 大数据处理

+ pig
+ Mapreduce
+ Spark,MLib

### 3.3 文本处理

+ TDIDF
+ word embeding
+ 词向量
+ 词袋模型
+ 互信息

### 3.4 图像处理


+ [图像金字塔](https://shartoo.github.io/image-pramid/)
+ SIFT特征
+ SURF特征
+ HOG特征
+ LAB特征
+ [Haar特征](https://shartoo.github.io/img-haar-feature/)
+ 轮廓
+ 边缘
+ 梯度
+ 角点检测
+ 模板匹配
+ [图像二值化方法](http://www.xmind.net/m/5Wgf)
<br>

+ [python opencv使用 xmind](http://www.xmind.net/m/dMye)
  - 图像变换
  - 几何变换
  - hough直线变换
  - hough圆变换
  - GrabGrabCut前景提取
  - 直方图
  - 图像轮廓
  - canny边缘检测
  - 形态学操作
  - 图像梯度
  - 图像平滑
  
+ [python opencv 特征提取 xmind](http://www.xmind.net/m/Ef6Z)
+ [python opencv 计算机摄影学 xmind](http://www.xmind.net/m/AEdb)
+ [python opencv 所有模块概览 xmind](http://www.xmind.net/m/xSD8)

+ 医学影像处理
 
### 3.5 视频分析

+ 背景消除
+ meanShift
+ CamShift
+ 光流

### 3.6 语音分析

1. 语音的基础概念

+ 概念：
+ 组成
  - 音素
  - 音节
  - 清音浊音
 
2. 语音相关特征

+ 频谱
+ 采样频率
+ 基音和泛音
+ 基频
+ 频域
+ 时域
+ 音高
+ 共振峰


3. 语音性质

+ 短时域
  - 短时能量
  - 短时平均幅度
  - 短时过零率
  - 短时自相关函数

+ 短频域
  - 傅里叶变换
 
4. 特征提取过程
 - A/D转换
 - 预加重
 - 加窗
   - Hanning窗
   - 矩形窗
 - DFT
 - Mel滤波
 - IDFT
 - 提取动态特征
 - 特征变换
 
 5. 语音对应的文本处理
 
 + 中文语音文本相关概念
  - 声母
  - 韵母
  - 韵母（转换后）

+ 处理过程

  1. 规范化
  2. 转化为拼音
  3. 拼音转音调
  4. 音节分解为音素
  
 
 6. 语音对应文本处理
 
 + 合成基元选取
 + 上下文相关标注
 + 问题集设计

## 4 数据挖掘/分析算法

### 4.1 机器学习算法

+ 决策树: ID3,C4.5,剪枝,熵，不纯度,回归树和分类树
+ 随机森林
+ 频繁模式挖掘：apiri算法，FP-growth算法
+ 线性回归
+ Logistic回归
+ KNN
+ K-mean：多种kmean
+ EM：[结合HMM算法](https://www.jianshu.com/p/a3572391a42d)
+ 贝叶斯分类
+ SVM分类
+ [集成学习](https://www.cnblogs.com/infaraway/p/7890558.html),[参考](https://www.cnblogs.com/willnote/p/6801496.html)
  -  [Boosting](https://www.cnblogs.com/willnote/p/6801496.html): AdaBoost,[GBDT](https://www.cnblogs.com/peizhe123/p/5086128.html),[XGBoost](https://blog.csdn.net/sb19931201/article/details/52557382)
  - Bagging: 随机森林
  - LightGBM:[原理](https://www.cnblogs.com/nxf-rabbit75/p/9748292.html),[代码](https://www.cnblogs.com/wanglei5205/p/8722237.html)

+ 线性判别分析
+ 主成分分析
+ EM算法
+ 遗传算法
+ 矩阵分解

### 4.2 Java Weka使用
### 4.3 Java Mahout使用
### 4.4 python sklean使用


+ <img src="https://github.com/shartoo/BeADataScientist/blob/master/data/images/sklearn_ml.svg?sanitize=true">

+ [sklearn 机器学习原图xmind](http://www.xmind.net/m/BNg7)


+ <img src="https://github.com/shartoo/BeADataScientist/blob/master/data/images/sklearn_feature_eng.svg?sanitize=true">
+ [sklearn 特征工程原图xmind](http://www.xmind.net/m/Jb2Y)

### 4.5 自然语言处理

+ 语言模型
+ 贝叶斯网络
+ 马尔科夫模型
+ 条件随机场
+ 命名实体识别
+ 词性标注
+ 语义分析
+ 句法分析
+ 情感分析
+ 搜索引擎

### 4.6 语音算法
 
#### 4.6.1 基础语音算法

+ GMM-HMM模型
+ DNN-HMM模型

#### 4.6.2 语音合成基础理论

[语音合成步骤](https://shartoo.github.io/texttospeech/)
[语音合成技术总结 xmind笔记](http://www.xmind.net/m/NMFv)

#### 4.6.3 开源语音合成框架


**merlin**

+ merlin 论文[原文](http://www.cstr.ed.ac.uk/downloads/publications/2016/Merlin_demo_paper.pdf)
+ merlin 理论体系
  - [merlin语音合成讲义一：技术路线概览](https://shartoo.github.io/merlin-tts-techmap1/)
  - [merlin语音合成讲义二：如何构建系统之数据准备](https://shartoo.github.io/merlin-tts-techmap2/)
  - [merlin语音合成讲义三：系统回归](https://shartoo.github.io/merlin-tts-techmap3/)

+ merlin 实战
  - [使用merlin从头构建你的声音](https://shartoo.github.io/merlin-tts/)
  - [使用merlin mandarin voice过程](https://shartoo.github.io/merlin-mandain-voice-op/)

+ merlin代码解析
  - [merlin使用 xmind](http://www.xmind.net/m/tbbW)
  - [merlin全局代码架构xmind](http://www.xmind.net/m/mRkg)
  - [merlin s1目录架构xmind](http://www.xmind.net/m/RgMa)
  - [merlin misc 代码架构 xmind](http://www.xmind.net/m/S3av)
  - [merlin src目录xmind](http://www.xmind.net/m/zkmS)
  - [merlin src-fronted目录 xmind](http://www.xmind.net/m/2bpe)
  - [merlin src-run_merlin.py 逻辑说明 xmind](http://www.xmind.net/m/fhZK)
  - [merlin-egs-mandarin-voice 概览 xmind](http://www.xmind.net/m/Fy9z)
  
  

#### 4.6.3 开源语音识别框架

+ 语音识别开源
   - kaldi

### 4.7 传统图像算法

+ [运动目标检测xmind](http://www.xmind.net/m/nk9G)
+ [python opencv中的机器学习算法](http://www.xmind.net/m/gvXh)
  - knn  ocr
  - svm ocr
  - kmeans

### 4.8 深度学习图像算法

**分类定位网络**
+ FasterRCNN
 - (RCNN,FastRCNN,SPP到FasterRCNN)[https://shartoo.github.io/RCNN-series/]
+ SSD 
   - [SSD 理解 medium](https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06)
   - [SSD 代码解析](https://shartoo.github.io/SSD_detail/)

**YOLO**: [从Yolov1到yolov3](https://shartoo.github.io/yolo-v123/)

**MobileNet**
 - MobileNetv1
 - MobileNetv2

 - RetinaNet

**人脸检测识别**

 - 人脸检测
 - 人脸识别

+ 图像分割算法
  - Unet,3DUnet
  - FCN
  - DeepLab
  - DenseASPP
  - ICNet
  - PSPNet
  - BiSeNet
 
+ 超分辨率
+ 图像生成算法
  - DCGAN
+ 看图说话
+ 风格迁移

### 4.9 传统图像算法

+ 图像分割
+ 特征检测
+ 追踪

+ [图像中的机器学习算法](http://www.xmind.net/m/nm89)
   - KNN
   - SVM
   - Kmeans
+ [python opencv 视频分析xmind](http://www.xmind.net/m/DA3x)

## 5 可视化展现

### 5.1  python 

+ matplotlib
+ scipy

### 5.2 Java 

+ D3
+ JFreeChart

### 5.3 关系网络可视化

+ java gephi

### 5.4 医疗图像可视化

+ Mevislab 
+ Mongo

### 5.5 神经网络网络可视化

+ Netron
+ TensorSpace
+ [caffe网络可视化](https://ethereon.github.io/netscope/#/editor)
+ [网络结构可视化 alexnet](http://scs.ryerson.ca/~aharley/vis/conv/)
+ [彩色网络架构可视化alexnet为例](http://alexlenail.me/NN-SVG/LeNet.html)
+ [网络架构可视化](https://github.com/HarisIqbal88/PlotNeuralNet)

## 6 数据搜索,推荐|搜索引擎设计，搜索系统，SEO，推荐系统

### 6.1 搜索引擎

+ Lucene使用
+ Lucene设计
+ ElasticSearch使用
+ ElasticSearch设计

### 6.2 搜索引擎设计原理
### 6.3 SEO优化简要
### 6.4 推荐系统

+ 常用算法
  - 协同过滤
  - 矩阵分解
  - FM
  - FFM
  - 聚类算法
  - SVD,SVD+
  - xgboost
  - 逻辑回归

+ 深度学习方法
  - Tensorflow deep and wide

+ 推荐系统冷启动问题


## 7 系统实现

### 7.1  Java web构建服务



