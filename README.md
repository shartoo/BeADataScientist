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

## 2 详细

### 2.0 数据源

#### 2.0.1 普通数据集

即被交付给的原始数据集

#### 2.0.2 公开数据集

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

#### 2.0.3 爬虫数据

+ HTML语言简介
+ python BeautifulSoup简介
+ scrapy简介

#### 2.1.1 kafka

+ kafka 简介
+ Java 调用kafka
+ kafka原理

#### 2.1.2 ZeroMQ

+ ZeroMQ 简介
+ ZeroMQ使用

### 2.2 数据储存

#### 2.2.1 SQL

+ SQL基础语法
+ SQL高级语法
+ SQL优化
+ SQL思考

#### 2.2.2 YARN

+ YARN生态简介

#### 2.2.3 HDFS

+ HDFS文件系统简介

#### 2.2.4 HBase

+ HBase 简介和安装
+ HBase java API简单调用
+ HBase 索引机制，二级索引设计
+ HBase 设计原理

#### 2.2.5 Hive

+ Hive简介和安装
+ Hive语法，存储
+ Hive设计

#### 2.2.6 MongoDB

+ MongoDB 简介和安装
+ MongoDB 设计

### 2.3 数据分析

#### 2.3.1 普通数据处理
#### 2.3.2 大数据处理

+ pig
+ Mapreduce
+ Spark,MLib

#### 2.3.3 文本处理

+ TDIDF
+ word embeding
+ 词向量
+ 词袋模型
+ 互信息

#### 2.3.4 图像处理


+ 图像金字塔
+ SIFT特征
+ SURF特征
+ HOG特征
+ LAB特征
+ Haar特征
+ 轮廓
+ 边缘
+ 梯度
+ 角点检测
+ 模板匹配

<br>

+ python opencv使用
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
  - 

+ 医学影像处理
 
#### 2.3.5 视频分析

+ 背景消除
+ meanShift
+ CamShift
+ 光流

#### 2.3.6 语音分析

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

### 2.4 数据挖掘/分析算法

#### 2.4.1 机器学习算法

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

#### 2.4.2 Java Weka使用
#### 2.4.3 Java Mahout使用
#### 2.4.4 python sklean使用
#### 2.4.5 自然语言处理

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

#### 2.4.6 语音识别
 
+ GMM-HMM模型
+ DNN-HMM模型

+ 语音合成开源
  - 传统方法：Merlin,Ossian
  - 深度学习方法: tacoron,wavenet,FloWaveNet
+ 语音识别开源
   - kaldi
  
#### 2.4.7 深度学习图像算法

+ 分类定位算法
 - FasterRCNN
 - SSD
 - Yolo1-3
 - MobileNet
 - RetinaNet

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

#### 2.4.8 传统图像算法

+ 图像分割
+ 特征检测
+ 追踪
+ 图像中的机器学习算法
   - KNN
   - SVM
   - Kmeans


### 2.5 可视化展现

#### 2.5.1  python 

+ matplotlib
+ scipy

#### 2.5.2 Java 

+ D3
+ JFreeChart

#### 2.5.3 关系网络可视化
+ java graphiv 

#### 2.5.4 医疗图像可视化

+ Mevislab 
+ Mongo

#### 2.5.5 神经网络网络可视化

+ Netron
+ TensorSpace
+ caffe网络可视化

### 2.6 数据搜索,推荐|搜索引擎设计，搜索系统，SEO，推荐系统

#### 2.6.1 搜索引擎

+ Lucene使用
+ Lucene设计
+ ElasticSearch使用
+ ElasticSearch设计

#### 2.6.2 搜索引擎设计原理
#### 2.6.3 SEO优化简要
#### 2.6.4 推荐系统

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


### 2.7 系统实现

#### 2.7.1  Java web构建服务



