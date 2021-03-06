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

大的数据收录网站

+ [github awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets)
+ [skymind open dataset](https://skymind.ai/wiki/open-datasets) : 保罗万象的数据集搜集网，什么类型的数据都有


### 0.1 普通数据集

即被交付给的原始数据集

### 0.2 公开数据集

#### 0.2.1 离散和连续型普通数据集
+ [uci数据集](http://archive.ics.uci.edu/ml/datasets.html) :一个公开的用于机器学习的数据集，包含几乎所有类别数据，适用于多种机器学习任务
+ 政府公开数据集 
  - [欧洲政府公开数据集](https://data.europa.eu/euodp/data/dataset)
  - [美国政府公开数据集](https://www.data.gov/)
  - [新西兰政府公开数据集](https://catalogue.data.govt.nz/dataset)
  - [印度政府公开数据集](https://data.gov.in/)
  - [中国人民银行,社会融资规模、金融统计数据、货币统计、金融机构信贷收支统计、金融市场统计、企业商品价格指数等](www.pbc.gov.cn/diaochatongjisi/116219/index.html)
  - [国内各类型银行业金融机构](www.cbrc.gov.cn/chinese/jrjg/index.html)
  - [中国国家统计局](www.stats.gov.cn/tjsj/)
  - [数据_中国政府网](www.gov.cn/shuju/)
  
#### 0.2.2 图像数据集

**图像分类**

+ [手写字识别MNIST，60000万张分辨率为28x28，数字0-9灰白图识别](http://yann.lecun.com/exdb/mnist/)
+ [CIFAR-10,6万张分辨率为32x32的10个分类的彩色分类图像](http://www.cs.toronto.edu/~kriz/cifar.html)
+ [ImageNet 1400万张，1000多个类别的分类图像，深度学习图像领域的关键比赛数据](http://www.image-net.org/)
+ [coco 微软组织的图像数据，包含了多种图像任务数据](http://cocodataset.org/)

**目标检测**

+ [coco 微软组织的图像数据，包含了多种图像任务数据](http://cocodataset.org/)
+ [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

**目标跟踪**

+ [OTB50和OTB100](https://link.jianshu.com/?t=http%3A%2F%2Fcvlab.hanyang.ac.kr%2Ftracker_benchmark%2F)
+ [VOT2013-2019](http://www.votchallenge.net/challenges.html)

**语义分割**

+ [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/):分割任务中VOC2012的trainval包含07-11年的所有对应图片， test只包含08-11。trainval有 2913张图片共6929个物体
+ [Cityscapes 城市道路分割](https://www.cityscapes-dataset.com/)：该数据集包含images_base和annotations_base分别对应这文件夹leftImg8bit（5,030 items, totalling 11.6 GB，factually 5000 items）和gtFine（30,030 items, totalling 1.1 GB）。里面都包含三个文件夹：train、val、test。总共5000张精细释，2975张训练图，500张验证图和1525张测试图。在leftImg8bit/train下有18个子文件夹对应德国的16个城市，法国一个城市和瑞士一个城市
+ [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)：城市街道分割
+ [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php)：移动机器人及自动驾驶研究的最受欢迎的数据集之一，包含7481张训练集，7518张测试集，总计80.256种标签。该网站也列出了历年的分割结果以及对应的方法。
+ [ADE20K_MIT](http://groups.csail.mit.edu/vision/datasets/ADE20K/)：场景理解的新的数据集包括各种物体（比如人、汽车等）、场景（天空、路面等），150个类别，22210张图。
+ [Sift Flow Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/): 包含2688张图片，33个labels，包括Awning（棚） balcony（阳台） bird（鸟） boat（船） bridge（桥）Building（建筑）等每一类都有百张左右。
+ [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html):包含从现有公共数据集中选择的715个图像，具有大约320×240像素，包含label种类：天空，树，道路，草，水，建筑物，山脉和前景物体.
+ [MSRC Dataset](https://pgram.com/dataset/msrc-v1/):240个图像，可识别9个object class。
+ [LIP](http://sysu-hcp.net/lip/):人体图像是从microsoft coco训练集和验证集中裁剪的。定义了19个人体部件或衣服标签，它们是帽子、头发、太阳镜、上衣、衣服、外套、袜子、裤子、手套、围巾、裙子、连体裤、脸、右臂、左臂、右腿、左腿、右脚、右脚鞋、左鞋，以及背景标签。数据集中共有50462张图像，其中包括19081张全身图像、13672张上身图像、403张下身图像、3386张头部丢失的图像、2778张后视图图像和21028张有遮挡的图像。
+ [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas?pKey=rwbBtYKofke2NeLIvj8j-A&lat=20&lng=0&z=1.5):25,000个高分辨率图像（分为18,000个用于训练，2,000个用于验证，5,000个用于测试）.152个物体类别，100个特定于实例的注释类别。一个多样化的街道级图像数据集，具有像素精确和特定于实例的人类注释，用于理解世界各地的街景。
+ [MIT SceneParse150](http://sceneparsing.csail.mit.edu/):MIT场景解析基准（SceneParse150）为场景解析算法提供标准的训练和评估平台。 该基准测试的数据来自ADE20K数据集。
+ [COCO 2017 Stuff Segmentation Challenge](https://cocodataset.org/#stuff-2019):COCO 2019 图像分割挑战赛。COCO数据集非常全面，可以从其[官方网站](https://cocodataset.org/#download)下载各类图像任务数据集。
+ [INRIA Annotations for Graz-02](https://lear.inrialpes.fr/people/marszalek/data/ig02/):2006年发布的数据集，包含人、自行车、汽车三类，合计超过2000张。
+ [Clothing Co-Parsing (CCP) Dataset](https://github.com/bearpaw/clothing-co-parsing):衣服分割图片，2098张高分辨率街头时尚照片，共59个标签。
+ [ApolloScape](http://apolloscape.auto/scene.html):百度提供的场景解析数据集,开放数据集累计提供146,997帧图像数据，总计34类，包含像素级标注和姿态信息，以及对应静态背景深度图像下载。

**图像融合**

+ [爱分割人脸matting数据集](https://github.com/aisegmentcn/matting_human_datasets):数据量大，包含34427张图像和对应的matting结果图，但是数据标注不够精细
+ [alpha matting官方数据集](http://www.alphamatting.com/datasets.php): 目前已知的最精细的数据集，但是数据量太少，只有27张。
+ [Deep Automatic Portrait Matting](http://xiaoyongshen.me/webpages/webpage_automatting/):包含2000张图像，精度和数量都适中，由于此数据集有很多明显的标注错误，需要进一步人工校正。

**超分辨率**

+ [Vimeo-90k](http://toflow.csail.mit.edu/):包含89800张从vimeo.com网站上视频截图，图像分辨率为448 x 256
+ [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)：22872张图像对，2D对应的3D图像，标注ground truth为光流。
+ [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)202599张各个尺寸的图片，此数据集有多类用途。
+ [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration/)：包含4741张原图，以及从这些图像中抖动生成的94,880图像。
+ [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/):包含800张训练集，100张验证集，100张测试集，总计1000张2k分辨率图像。

**人脸数据集**

+ [CARC](https://bcsiriuschen.github.io/CARC/):[百度网盘](https://pan.baidu.com/s/1HbuOtTyj2vQaok3hXkl_6g) 提取码 dvyn 
+ celaba
  - [celaba 128x128分辨率原始数据集 官方下载](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [celaba 128x128分辨率原始数据集 百度云下载](https://pan.baidu.com/s/1eSNpdRG#list/path=%2F)
  - [celaba hq 数据集生成方法](https://zhuanlan.zhihu.com/p/52188519)
  - [celaba hq 数据集图片格式 128x128,256x256,512x512,1024x1024 谷歌drive下载](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P) 
  - [celaba hq 官方提供的dat格式下载](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs)
 
 + **年龄和表情数图像据集**
   - [FACES](https://faces.mpdl.mpg.de/imeji/):包含了 171 young (n = 58), middle-aged (n = 56), and older (n = 57) ，不同表情的图片数据集。
  
 + **医疗图像数据集**

#### 0.2.3 语音数据
   - 语音识别数据集
   - 语音合成数据集

#### 0.2.4 视频数据集
#### 0.2.5 金融股票数据集

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

#### 3.1.1 数据清洗
#### 3.1.2 数据标准化
#### 3.1.3 数据映射(降维/升维)

**降维**

1. 线性判别分析
  + [博客](https://www.cnblogs.com/pinard/p/6244265.html)
  + 代码
2. 二次判别分析
 + [博客](https://www.cnblogs.com/xingshansi/p/6892317.html)
 + 代码

3. 矩阵分解-PCA
  + [博客](https://zhuanlan.zhihu.com/p/32412043)
  + 关键: 用数据里最主要的方面来代替原始数据。连续数据。无监督
4. 矩阵分解-kernelPCA
  + [博客](https://zhuanlan.zhihu.com/p/25097144)
  + 关键: PCA在协方差上变换，kernel PCA在kernel矩阵上变换
5. 矩阵分解-稀疏PCA
  + [博客](https://blog.csdn.net/zhoudi2010/article/details/53489319)
  + 关键: 解决稀疏数据降维
6. 随机投影-高斯随机投影
7.流型学习-MDS
8. [流型学习-ISOMap](https://www.cnblogs.com/wing1995/p/5479036.html)
9. [流型学习-LocallyLinearEmbedding](https://www.cnblogs.com/pinard/p/6273377.html)
10. [流型学习-拉普拉斯特征映射LE算法](https://hanyuz1996.github.io/2017/08/20/Laplacian%20Eigenmap/)


**升维**

1. [流型学习-tSNE](https://www.jiqizhixin.com/articles/2017-11-13-7)

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
+ [SIFT特征](https://shartoo.github.io/SIFT-feature/)
+ SURF特征
+ [HOG特征](https://shartoo.github.io/HOG-feature/)
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

#### 4.1.1 分类算法

+ 决策树: ID3,C4.5,剪枝,熵，不纯度,回归树和分类树
+ 贝叶斯分类
+ SVM分类
+ 线性回归
+ Logistic回归
+ KNN

#### 4.1.2 聚类算法

+ K-mean：多种kmean
+ 矩阵分解

#### 4.1.2 数据压缩算法

+ 主成分分析

#### 4.1.3 其他

+ 随机森林
+ 频繁模式挖掘：apiri算法，FP-growth算法


+ EM：[结合HMM算法](https://www.jianshu.com/p/a3572391a42d)

+ [集成学习](https://www.cnblogs.com/infaraway/p/7890558.html),[参考](https://www.cnblogs.com/willnote/p/6801496.html)
  -  [Boosting](https://www.cnblogs.com/willnote/p/6801496.html): AdaBoost,[GBDT](https://www.cnblogs.com/peizhe123/p/5086128.html),[XGBoost](https://blog.csdn.net/sb19931201/article/details/52557382)
  - Bagging: 随机森林
  - LightGBM:[原理](https://www.cnblogs.com/nxf-rabbit75/p/9748292.html),[代码](https://www.cnblogs.com/wanglei5205/p/8722237.html)

+ 线性判别分析
+ EM算法
+ 遗传算法

#### 4.1.4 统计学模型

+ [时间序列模型](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)

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
+ FasterRCNN: [RCNN FastRCNN SPP到FasterRCNN](https://shartoo.github.io/RCNN-series/)
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
 - 人脸检测识别项目
    - [中科大山世光开源C++人脸检测识别 seetface 代码](http://www.xmind.net/m/wUGM)

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



