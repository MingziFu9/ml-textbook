---
title: 附录 D 术语表
description: 本书主要术语的中英文名称、简明定义和例子，按章节顺序排列
---

# 附录 D 术语表

本表按章节顺序列出本书的主要术语，每个术语给出英文名称、简明定义和例子。正文中带虚线下划线的术语，把鼠标停在上面就能看到定义，点击可以跳到本表。

通用术语（如“过拟合”“精确率”）的译名和定义以本系列教材共用的《系列公共术语表》为准，各书保持一致。

其余各章的术语随正文编写陆续补入。

## 第 1 章 机器学习概述与项目流程

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-机器学习"></span>机器学习 | Machine Learning | 让计算机从数据中自动找出规律，并用这些规律对新数据做出判断或预测的方法 | 从十万条已标注的短信中学会识别垃圾短信 |
| <span id="g-深度学习"></span>深度学习 | Deep Learning | 使用多层神经网络从数据中自动学习特征的机器学习方法 | 图像识别、语音识别和大模型都基于深度学习 |
| <span id="g-样本"></span>样本 | Sample | 数据中的一个个体或一条记录，是机器学习的基本单位 | 鸢尾花数据中的一朵花、客户表中的一个客户 |
| <span id="g-特征"></span>特征 | Feature | 描述样本的属性，通常整理成数字，作为模型的输入 | 花瓣长度、月收入、合约类型 |
| <span id="g-标签"></span>标签 | Label | 样本的正确答案，也就是希望模型学会预测的那个量 | 鸢尾花的品种、客户是否流失 |
| <span id="g-数据集"></span>数据集 | Dataset | 由许多样本组成、用于训练或评估模型的数据集合 | 鸢尾花数据集有 150 个样本、4 个特征 |
| <span id="g-模型"></span>模型 | Model | 机器学习从数据中学到的、能把特征转换成预测结果的规律 | 一棵决策树、一个线性回归公式 |
| <span id="g-模型参数"></span>模型参数 | Model Parameter | 模型中由训练从数据里确定的数值 | 线性回归的权重和偏置、决策树每个节点用的特征和门槛 |
| <span id="g-训练"></span>训练 | Training | 用数据确定模型参数的过程，也叫学习或拟合 | `model.fit(X_train, y_train)` |
| <span id="g-预测"></span>预测 | Prediction | 把新样本的特征输入训练好的模型，得到它的标签 | `model.predict([[5.0, 3.4, 1.5, 0.2]])` 判为山鸢尾 |
| <span id="g-泛化能力"></span>泛化能力 | Generalization Ability | 模型在没见过的新数据上也能表现良好的能力，是评价模型最重要的标准 | 决策树训练集准确率 0.857、新申请上 0.828 |
| <span id="g-训练集"></span>训练集 | Training Set | 用来训练模型的数据，相当于“课本和练习题” | 鸢尾花 150 朵中随机抽取的 120 朵 |
| <span id="g-测试集"></span>测试集 | Test Set | 训练时不让模型看见、训练完成后用来评估模型的数据，相当于“期末考试卷” | 剩下的 30 朵花 |
| <span id="g-验证集"></span>验证集 | Validation Set | 训练过程中用来调整参数、选择阈值的数据，相当于“模拟考试” | 从训练数据中再分出 20% 用来比较不同的最大深度 |
| <span id="g-监督学习"></span>监督学习 | Supervised Learning | 从带有正确答案（标签）的样本中学习规律的机器学习方式 | 用标好“流失 / 不流失”的历史客户数据训练流失预测模型 |
| <span id="g-回归"></span>回归 | Regression | 预测一个连续数值的监督学习任务，如预测销量、房价、气温 | 预测二手房总价、外卖送达时间 |
| <span id="g-分类"></span>分类 | Classification | 预测样本属于预先定义好的哪个类别的监督学习任务 | 判断客户是否流失、识别手写数字 0～9 |
| <span id="g-无监督学习"></span>无监督学习 | Unsupervised Learning | 训练数据没有标签，由模型自己发现数据中的结构或规律的机器学习方式 | 聚类、降维、关联规则、异常检测 |
| <span id="g-强化学习"></span>强化学习 | Reinforcement Learning | 智能体在环境中不断尝试、根据奖励调整行为，学习最优行动策略的机器学习方式 | AlphaGo 通过自我对弈提高棋力 |
| <span id="g-半监督学习"></span>半监督学习 | Semi-supervised Learning | 同时利用少量有标签数据和大量无标签数据训练模型的方法 | 十万张影像中只有一千张经过专家标注 |
| <span id="g-自监督学习"></span>自监督学习 | Self-supervised Learning | 从数据本身自动构造标签来训练模型的方式，大模型的预训练就是一例 | 遮住句子中的一个词让模型猜，大模型预训练用的就是这种方法 |
| <span id="g-机器学习项目流程"></span>机器学习项目流程 | Machine Learning Project Workflow | 问题定义、数据准备、特征工程、建模、评估、部署与监控六个步骤反复迭代的工作过程，参考 CRISP-DM | 客户流失预警项目从定义“未来一个月是否销号”开始 |
| <span id="g-scikit-learn"></span>scikit-learn | scikit-learn | 最流行的 Python 经典机器学习库，所有模型都用 fit、predict、score 统一接口 | `from sklearn.neighbors import KNeighborsClassifier` |
| <span id="g-超参数"></span>超参数 | Hyperparameter | 训练前由人设定、不由模型学习的参数 | K 近邻的邻居数 `n_neighbors=5`、决策树的 `max_depth` |

## 第 2 章 数据预处理与特征工程

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-数据质量"></span>数据质量 | Data Quality | 数据满足使用要求的程度，常从完整性、准确性、一致性、时效性等方面评价 | 3060 行二手房数据中有缺失、重复、格式不一致和录入错误 |
| <span id="g-探索性数据分析"></span>探索性数据分析 | Exploratory Data Analysis, EDA | 在建模之前通过统计和画图了解数据分布、关系和问题的过程 | 用 `describe()`、直方图、箱线图检查二手房数据 |
| <span id="g-描述性统计"></span>描述性统计 | Descriptive Statistics | 用均值、中位数、标准差、四分位数等少数几个数概括一列数据 | 总价均值 5216.2 万元、中位数 243.0 万元，说明有极端值 |
| <span id="g-数据透视"></span>数据透视 | Pivot Table | 按一个或几个类别分组统计，并排成交叉表格的分析方法 | 各区不同装修的单价中位数表 |
| <span id="g-相关系数"></span>相关系数 | Correlation Coefficient | 衡量两个数值变量线性关系强弱和方向的指标，取值在 −1 到 1 之间 | 单价与总价的相关系数为 0.77 |
| <span id="g-数据清洗"></span>数据清洗 | Data Cleaning | 发现并处理数据中的缺失、重复、错误和不一致，使数据准确、完整、一致的过程 | 去重、去单位、修正按“元”录入的总价 |
| <span id="g-缺失值"></span>缺失值 | Missing Value | 数据中应该有却没有记录的值 | 213 套房子没有填建成年份 |
| <span id="g-异常值"></span>异常值 | Outlier | 明显偏离大多数数据的值，可能是错误，也可能是真实但少见的情况 | 面积 1295 平方米（录入错误）；天河 1286.7 万元的大平层（真实） |
| <span id="g-3σ-法则"></span>3σ 法则 | Three-sigma Rule | 把落在“均值 ± 3 倍标准差”之外的值视为异常的规则，适用于近似正态分布的数据 | 清洗后总价超过 761.7 万元的 45 套 |
| <span id="g-箱线图法"></span>箱线图法 | IQR Method | 把落在 [Q1 − 1.5 × IQR, Q3 + 1.5 × IQR] 之外的值视为异常的规则，IQR = Q3 − Q1 | 清洗后总价超过 652.0 万元的 106 套 |
| <span id="g-重复值"></span>重复值 | Duplicate | 同一个对象在数据中被记录了多次 | 同一房源编号出现两次的 60 套房 |
| <span id="g-数据清洗规范"></span>数据清洗规范 | Data Cleaning Specification | 写明每个字段的问题、处理规则、理由和影响条数的书面文件，使清洗过程可检查、可重复 | 二手房数据清洗规范 R1～R8 |
| <span id="g-数值型特征"></span>数值型特征 | Numerical Feature | 取值是有大小意义的数的特征 | 面积、建成年份、距地铁站 |
| <span id="g-类别型特征"></span>类别型特征 | Categorical Feature | 取值是有限几个类别的特征 | 区域、装修、朝向 |
| <span id="g-标签编码"></span>标签编码 | Label Encoding | 给每个类别分配一个整数的编码方法，只适合本身有顺序的类别 | 低楼层、中楼层、高楼层编为 0、1、2 |
| <span id="g-独热编码"></span>独热编码 | One-Hot Encoding | 把一个有 k 个取值的类别特征变成 k 列 0/1，每个样本只在所属类别一列为 1 | “装修”变成装修_精装、装修_简装、装修_毛坯、装修_未知四列 |
| <span id="g-高基数类别特征"></span>高基数类别特征 | High-cardinality Categorical Feature | 取值非常多的类别特征，直接独热编码会产生大量稀疏的列 | 337 个小区、上万个商品编号 |
| <span id="g-目标编码"></span>目标编码 | Target Encoding | 用每个类别对应的标签平均值（或中位数）代替类别的编码方法，必须只用训练集计算并做平滑 | 用“小区参考单价”代替小区名 |
| <span id="g-特征缩放"></span>特征缩放 | Feature Scaling | 把不同特征的数值调整到可比范围的处理 | 标准化、归一化 |
| <span id="g-标准化"></span>标准化 | Standardization | 减去均值、除以标准差，使特征均值为 0、标准差为 1 | 63.5 平方米标准化后为 −0.94 |
| <span id="g-归一化"></span>归一化 | Min-Max Normalization | 减去最小值、除以“最大值 − 最小值”，使特征取值在 0 到 1 之间 | 63.5 平方米归一化后约为 0.148 |
| <span id="g-特征工程"></span>特征工程 | Feature Engineering | 把原始数据加工成对模型有用的特征的全部工作，包括编码、缩放、构造和选择 | 从挂牌日期算房龄，独热编码区域 |
| <span id="g-特征构造"></span>特征构造 | Feature Construction | 根据业务理解从已有字段计算出新特征 | 每室面积 = 面积 ÷ 室数 |
| <span id="g-日期特征"></span>日期特征 | Date Feature | 从日期中拆出的年、月、星期、节假日或时间差等特征 | 房龄 = 挂牌年份 − 建成年份 |
| <span id="g-交叉特征"></span>交叉特征 | Cross Feature | 由两个或多个特征组合（如相乘）得到的新特征 | 参考总价 = 区域参考单价 × 面积，测试 R² 从 0.839 升到 0.903 |
| <span id="g-分箱"></span>分箱 | Binning | 把连续数值按区间分成若干段，变成类别特征 | 面积分成小户型、中户型、大户型 |
| <span id="g-特征选择"></span>特征选择 | Feature Selection | 从已有特征中挑出最有用的一部分的过程 | 从 33 个特征中选出 10 个 |
| <span id="g-过滤法"></span>过滤法 | Filter Method | 训练模型之前按统计指标给特征打分并筛选的特征选择方法 | 按与总价的相关系数选前 10 个 |
| <span id="g-包裹法"></span>包裹法 | Wrapper Method | 把特征子集放进模型中试、按模型成绩挑选子集的特征选择方法 | 递归特征消除（RFE） |
| <span id="g-嵌入法"></span>嵌入法 | Embedded Method | 在模型训练过程中自动完成特征选择的方法 | Lasso 把 6 个特征的系数压为 0 |
| <span id="g-数据泄露"></span>数据泄露 | Data Leakage | 训练时用到了真实预测时刻无法获得的信息，导致评估成绩虚高的错误 | 预测总价时用了由总价算出的单价 |

## 第 3 章 线性回归与梯度下降

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-线性回归"></span>线性回归 | Linear Regression | 假设标签等于各特征的加权和再加一个常数的回归模型 | 二手房多元线性回归测试集 R² 为 0.836 |
| <span id="g-一元线性回归"></span>一元线性回归 | Simple Linear Regression | 只有一个特征的线性回归，模型是一条直线 ŷ = wx + b | 5 套房子拟合出 总价 = 2.55 × 面积 + 30.5 |
| <span id="g-多元线性回归"></span>多元线性回归 | Multiple Linear Regression | 有多个特征的线性回归，每个特征有自己的权重 | 用面积、区域、房龄等 25 个特征估计二手房总价 |
| <span id="g-权重与偏置"></span>权重与偏置 | Weight and Bias | 线性模型中由训练学到的参数：权重是特征的系数，偏置是常数项 | w = 2.55 表示每多 1 平方米总价多 2.55 万元，b = 30.5 |
| <span id="g-损失函数"></span>损失函数 | Loss Function | 衡量模型预测与真实答案之间差距的函数 | 分类用交叉熵，预测房价用均方误差 |
| <span id="g-残差"></span>残差 | Residual | 一个样本的真实值减去预测值 | 5 套房子的残差为 2、1、−10、9、−2 |
| <span id="g-均方误差"></span>均方误差 | Mean Squared Error, MSE | 残差平方的平均值，是线性回归最常用的损失 | 5 套房子的残差平方和 190，MSE = 38 |
| <span id="g-最小二乘法"></span>最小二乘法 | Least Squares | 让残差平方和最小来求模型参数的方法，一元线性回归有现成公式 | w = 10200 / 4000 = 2.55，b = 260 − 2.55 × 90 = 30.5 |
| <span id="g-梯度"></span>梯度 | Gradient | 损失函数对参数的偏导数，指示参数调整的方向和幅度 | 梯度为正，说明应该把这个权重调小 |
| <span id="g-梯度下降"></span>梯度下降 | Gradient Descent | 沿梯度反方向更新参数以减小损失的优化方法 | 蒙眼下山，每次往最陡的下坡方向迈一小步 |
| <span id="g-学习率"></span>学习率 | Learning Rate | 每次参数更新的步长，是最重要的超参数之一 | 5 套房子上学习率 0.5 二十步收敛，1.05 发散 |
| <span id="g-随机梯度下降"></span>随机梯度下降 | Stochastic Gradient Descent, SGD | 每次只用一个随机样本计算梯度并更新参数的梯度下降 | 2400 个样本训练 20 轮，共更新 48000 次 |
| <span id="g-小批量梯度下降"></span>小批量梯度下降 | Mini-batch Gradient Descent | 每次用一小批随机样本计算梯度并更新参数的梯度下降，最常用 | 每批 32 个样本，20 轮后 MSE 17680.3，最接近最优值 17606.1 |
| <span id="g-多重共线性"></span>多重共线性 | Multicollinearity | 特征之间高度相关，导致回归系数不稳定、难以解释的现象 | “室”与“面积”相关系数 0.845，“室”的系数从 75.30 变成 2.49 |
| <span id="g-平均绝对误差"></span>平均绝对误差 | Mean Absolute Error, MAE | 残差绝对值的平均，单位与标签相同 | 二手房线性回归平均每套估错 47.9 万元 |
| <span id="g-均方根误差"></span>均方根误差 | Root Mean Squared Error, RMSE | 均方误差的平方根，单位与标签相同，对大误差更敏感 | 二手房线性回归 RMSE 66.2 万元，大于 MAE 47.9 万元 |
| <span id="g-决定系数"></span>决定系数 | Coefficient of Determination, R² | 用 1 减去模型与“猜平均值”的误差平方和之比，1 为完美，0 与猜平均值相当 | 猜平均值 R² 为 0，线性回归 0.836，预测 log(总价) 0.910 |
| <span id="g-残差图"></span>残差图 | Residual Plot | 以预测值为横轴、残差为纵轴的散点图，用来发现模型漏掉的规律 | 线性回归的残差图呈喇叭口，高价房误差大 |
| <span id="g-多项式回归"></span>多项式回归 | Polynomial Regression | 把特征的平方、立方等各次方也作为特征的线性回归，可以拟合曲线 | 用气温的 1～15 次方预测奶茶销量 |
| <span id="g-欠拟合"></span>欠拟合 | Underfitting | 模型太简单、没学到规律，训练集和测试集成绩都差的现象 | 用直线拟合先升后降的销量，训练、测试 MSE 为 594.4、640.0 |
| <span id="g-过拟合"></span>过拟合 | Overfitting | 模型在训练集上表现好、在新数据上表现差的现象 | 15 阶多项式训练 MSE 155.8，测试 MSE 四千多万 |
| <span id="g-正则化"></span>正则化 | Regularization | 在损失函数中加入对参数大小的惩罚，以减轻过拟合的方法 | 加岭回归后系数绝对值之和从 165 万降到 71.3 |
| <span id="g-岭回归"></span>岭回归 | Ridge Regression | 在均方误差上加系数平方和（L2）惩罚的线性回归，系数整体变小但不为 0 | 15 阶多项式加岭回归（α = 1），测试 MSE 降到 512.5 |
| <span id="g-Lasso-回归"></span>Lasso 回归 | Lasso Regression | 在均方误差上加系数绝对值之和（L1）惩罚的线性回归，能把部分系数压成 0 | α = 1 时 15 个系数中 12 个变为 0 |

## 第 4 章 逻辑回归与分类模型评估

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-二分类"></span>二分类 | Binary Classification | 只有两个类别的分类任务 | 客户流失 / 不流失，肿瘤恶性 / 良性 |
| <span id="g-正类"></span>正类 | Positive Class | 二分类中要找出来、记为 1 的那一类，通常是少数、“出了问题”的一类 | 流失客户、恶性肿瘤、欺诈交易 |
| <span id="g-多分类"></span>多分类 | Multi-class Classification | 有三个及以上类别的分类任务 | 识别手写数字是 0～9 中的哪一个 |
| <span id="g-逻辑回归"></span>逻辑回归 | Logistic Regression | 给每个特征一个权重、加权求和后用 Sigmoid 函数输出类别概率的分类模型 | 电信客户流失模型测试集 AUC 为 0.836 |
| <span id="g-Sigmoid-函数"></span>Sigmoid 函数 | Sigmoid Function | 把任意实数压缩到 0 到 1 之间的 S 形函数 σ(z) = 1 / (1 + e⁻ᶻ) | σ(0) = 0.5，σ(2) ≈ 0.881，σ(−2) ≈ 0.119 |
| <span id="g-概率输出"></span>概率输出 | Probability Output | 分类模型给出的样本属于某一类的概率，而不只是类别 | 客户 C101053 的流失概率为 0.449 |
| <span id="g-分类阈值"></span>分类阈值 | Classification Threshold | 把概率转换成类别时使用的分界值 | 恶性概率大于等于 0.3 判为恶性，漏诊 3 例 |
| <span id="g-决策边界"></span>决策边界 | Decision Boundary | 分类模型把不同类别分开的分界线（面）；逻辑回归中是 z = 0 的直线或平面 | 两个特征时，直线一侧判为流失、另一侧判为不流失 |
| <span id="g-交叉熵"></span>交叉熵 | Cross-Entropy | 分类任务常用的损失函数，等于正确类别预测概率的负对数 | 正确类别概率 0.9 时损失约 0.105，0.1 时约 2.303 |
| <span id="g-几率"></span>几率 | Odds | 事件发生的概率与不发生的概率之比；逻辑回归中特征加 1，几率变为 e^w 倍 | 流失概率 0.75 时几率为 3；签两年合约使流失几率降为 0.47 倍 |
| <span id="g-Softmax"></span>Softmax | Softmax | 把一组得分转换为和为 1 的概率分布的函数 | 得分 2.0、1.0、0.1 → 概率 0.66、0.24、0.10 |
| <span id="g-混淆矩阵"></span>混淆矩阵 | Confusion Matrix | 按“实际类别 × 预测类别”交叉统计样本数量的表格 | 流失模型 TP = 264，FN = 191，FP = 147，TN = 898 |
| <span id="g-准确率"></span>准确率 | Accuracy | 预测正确的样本占全部样本的比例 | 1500 个客户中判对 1162 个，准确率约 0.775 |
| <span id="g-精确率"></span>精确率 | Precision | 预测为某类的样本中，真正属于该类的比例（判得准不准） | 判为流失的 411 人中，264 人真的流失，精确率 0.642 |
| <span id="g-召回率"></span>召回率 | Recall | 某类的全部样本中，被正确找出来的比例（找得全不全） | 455 个流失客户中找出 264 个，召回率 0.580 |
| <span id="g-F1-值"></span>F1 值 | F1 Score | 精确率与召回率的调和平均数，综合衡量两者 | 精确率 0.642、召回率 0.580 时 F1 约 0.610 |
| <span id="g-假阳性率"></span>假阳性率 | False Positive Rate, FPR | 实际为负类的样本中被误判为正类的比例，是 ROC 曲线的横轴 | 107 例良性中误诊 9 例，假阳性率 0.084 |
| <span id="g-ROC-曲线"></span>ROC 曲线 | Receiver Operating Characteristic Curve | 以假阳性率为横轴、真阳性率（召回率）为纵轴，把所有阈值的结果连成的曲线 | 好的模型曲线迅速升到左上角 |
| <span id="g-AUC"></span>AUC | Area Under the ROC Curve | ROC 曲线下的面积，等于随机正类样本得分高于随机负类样本的概率，0.5 为乱猜、1 为完美 | 乳腺肿瘤 5 个特征模型 AUC 0.980，30 个特征 0.998 |
| <span id="g-PR-曲线"></span>PR 曲线 | Precision-Recall Curve | 以召回率为横轴、精确率为纵轴的曲线，正类很少时比 ROC 更能说明问题 | 欺诈模型 AUC 0.962，PR 曲线下面积（AP）只有 0.487 |
| <span id="g-类别不平衡"></span>类别不平衡 | Class Imbalance | 各类别样本数量相差很大的情况 | 2 万笔交易中欺诈只占 1.04% |
| <span id="g-类别权重"></span>类别权重 | Class Weight | 在损失函数中给少数类样本更大权重的不平衡处理方法 | `class_weight="balanced"` 让召回率从 0.270 升到 0.841 |
| <span id="g-过采样与欠采样"></span>过采样与欠采样 | Oversampling and Undersampling | 增加少数类或减少多数类训练样本，使各类数量接近的方法，只能用于训练集 | 随机欠采样后召回率 0.873，误报 813 笔 |
| <span id="g-SMOTE"></span>SMOTE | Synthetic Minority Over-sampling Technique | 在少数类样本与其近邻之间插值、合成新样本的过采样方法 | SMOTE 过采样后召回率 0.841，误报 638 笔 |
| <span id="g-错误案例分析"></span>错误案例分析 | Error Analysis | 逐条查看模型判错的样本，归纳出错原因并提出改进建议的方法 | 漏判的欺诈多为小额、无设备变化的交易 |
| <span id="g-模型测试报告"></span>模型测试报告 | Model Test Report | 说明测试目的、数据、指标、结果、错误分析和结论建议，作为模型能否上线依据的文档 | 信用卡欺诈模型测试报告推荐类别权重 + 阈值 0.8 |

## 第 5 章 决策树与集成学习

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-决策树"></span>决策树 | Decision Tree | 用一连串是 / 否问题对样本逐步划分，最后得出类别或数值的模型 | “有房吗？→ 有工作吗？”两层问题决定是否批准贷款 |
| <span id="g-根节点"></span>根节点 | Root Node | 决策树最上面的节点，所有样本都从这里进入 | 贷款决策树的根节点是“有自己的房子？” |
| <span id="g-叶节点"></span>叶节点 | Leaf Node | 决策树中不再提问、直接给出预测结果的节点 | “批准”“不批准”两个叶节点 |
| <span id="g-树的深度"></span>树的深度 | Tree Depth | 从根节点到最远叶节点要经过的提问层数 | 两层问题的审批树深度为 2；`max_depth=3` 限制深度不超过 3 |
| <span id="g-信息熵"></span>信息熵 | Information Entropy | 衡量一组样本类别混乱程度的指标，越纯越小，二分类时取值在 0 到 1 之间 | 9 个批准、6 个不批准的数据集熵约为 0.971 |
| <span id="g-信息增益"></span>信息增益 | Information Gain | 按某个特征划分后信息熵减少的量，越大说明这个特征越能区分类别 | “有自己的房子”的信息增益为 0.420，四个特征中最大 |
| <span id="g-ID3-算法"></span>ID3 算法 | ID3 Algorithm | 昆兰 1986 年提出、用信息增益选择划分特征的决策树算法 | 15 条贷款数据按 ID3 先选“有自己的房子” |
| <span id="g-信息增益率"></span>信息增益率 | Information Gain Ratio | 信息增益除以特征固有值的结果，用来纠正信息增益偏爱取值多的特征 | “编号”的信息增益率只有 0.249，低于“有自己的房子”的 0.433 |
| <span id="g-C4.5-算法"></span>C4.5 算法 | C4.5 Algorithm | 昆兰 1993 年提出、用信息增益率选择特征，能处理连续特征和缺失值并带剪枝的决策树算法 | 数据挖掘十大算法之一 |
| <span id="g-基尼系数"></span>基尼系数 | Gini Impurity | 随机抽两个样本、它们类别不同的概率，用来衡量节点的不纯度 | 8 个批准、2 个不批准的节点基尼系数为 0.32 |
| <span id="g-CART-算法"></span>CART 算法 | Classification and Regression Tree, CART | 布雷曼等人 1984 年提出、用基尼系数或均方误差划分、只生成二叉树的决策树算法 | scikit-learn 的 `DecisionTreeClassifier` 采用优化的 CART |
| <span id="g-回归树"></span>回归树 | Regression Tree | 叶节点输出数值、用误差平方和选择划分的决策树 | 面积 ≤ 90 平方米的房子预测单价 2.1 万元 |
| <span id="g-剪枝"></span>剪枝 | Pruning | 通过限制或删减分支来降低决策树复杂度、减轻过拟合的方法 | 把不限深度的树剪到 6 层，测试集准确率从 0.773 升到 0.827 |
| <span id="g-预剪枝"></span>预剪枝 | Pre-pruning | 在树生长过程中提前停止划分的剪枝方法 | 设置 `max_depth`、`min_samples_leaf` |
| <span id="g-后剪枝"></span>后剪枝 | Post-pruning | 先让树充分生长，再自下而上剪掉对泛化无益的子树的剪枝方法 | CART 的代价复杂度剪枝，参数 `ccp_alpha` |
| <span id="g-集成学习"></span>集成学习 | Ensemble Learning | 把多个模型的预测组合起来，得到比单个模型更好结果的方法 | 3 个准确率 70% 的独立分类器投票，准确率升到 78.4% |
| <span id="g-Bagging"></span>Bagging | Bagging (Bootstrap Aggregating) | 用自助采样得到多份训练集、并行训练多个模型再投票或平均的集成方法，主要降低方差 | 随机森林是在 Bagging 基础上发展来的 |
| <span id="g-Boosting"></span>Boosting | Boosting | 串行训练多个弱学习器、后一个重点纠正前面的错误、最后加权组合的集成方法，主要降低偏差 | AdaBoost、梯度提升树、XGBoost |
| <span id="g-自助采样"></span>自助采样 | Bootstrap Sampling | 从 n 个样本中有放回地抽取 n 次得到新样本集的方法，约 63.2% 的不同样本会被抽中 | 每棵树约有 36.8% 的样本没有见过 |
| <span id="g-随机森林"></span>随机森林 | Random Forest | 对自助采样的数据训练多棵决策树，并在每个节点随机选部分特征划分，最后投票或平均的集成模型 | 300 棵树的随机森林预测客户流失，测试集 AUC 0.819 |
| <span id="g-袋外误差"></span>袋外误差 | Out-of-Bag Error | 用每个样本没参与训练的那些树来预测它而得到的错误率，可代替验证集估计效果 | 客户流失随机森林的袋外准确率为 0.798 |
| <span id="g-特征重要性"></span>特征重要性 | Feature Importance | 衡量各个特征对模型预测贡献大小的指标，树模型中常按不纯度下降量累计 | 在网月数和合约类型是影响流失最重要的两个特征 |
| <span id="g-弱学习器"></span>弱学习器 | Weak Learner | 单独使用时只比随机猜测略好的简单模型，是 Boosting 的组成单元 | 只有一层的决策树桩 |
| <span id="g-AdaBoost"></span>AdaBoost | AdaBoost (Adaptive Boosting) | 每轮调高被错分样本的权重、按错误率给各弱学习器分配话语权的 Boosting 算法 | 错误率 0.3 的弱学习器话语权约为 0.42 |
| <span id="g-梯度提升树"></span>梯度提升树 | Gradient Boosting Decision Tree, GBDT | 每棵新树拟合当前模型的残差（负梯度），按学习率累加的 Boosting 模型 | 起点预测 140 万元，第一棵树把残差平方和从 5600 降到 1550 |
| <span id="g-残差"></span>残差 | Residual | 真实值减去当前预测值得到的差 | 真实价 100 万元、预测 140 万元，残差为 −40 万元 |
| <span id="g-早停"></span>早停 | Early Stopping | 在验证集指标连续若干轮不再改善时停止训练的方法 | 学习率 0.1 的 XGBoost 在第 49 棵树时早停 |
| <span id="g-XGBoost"></span>XGBoost | XGBoost (eXtreme Gradient Boosting) | 陈天奇等人开发的梯度提升树实现，加入正则化、二阶近似和缺失值自动处理 | `xgb.XGBClassifier(max_depth=3, learning_rate=0.1)` |
| <span id="g-LightGBM"></span>LightGBM | LightGBM (Light Gradient Boosting Machine) | 微软亚洲研究院开发的梯度提升树实现，用直方图算法和按叶子生长提高速度、节省内存 | `lgb.LGBMClassifier(num_leaves=8)` |

## 第 6 章 支持向量机

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-支持向量机"></span>支持向量机 | Support Vector Machine, SVM | 在两类样本之间寻找间隔最大的分界线的分类算法 | 手写数字识别测试准确率 0.989，540 张只认错 6 张 |
| <span id="g-线性可分"></span>线性可分 | Linearly Separable | 存在一条直线（平面）能把两类样本完全分开 | 同心圆数据线性不可分，加上 r² 特征后变得线性可分 |
| <span id="g-间隔"></span>间隔 | Margin | SVM 中两条间隔边界 f(x) = ±1 之间的距离，等于 2/‖w‖ | w = (0.5, 0.5) 时间隔宽度 2 / 0.707 ≈ 2.83 |
| <span id="g-支持向量"></span>支持向量 | Support Vector | 落在间隔边界上或间隔以内（含被分错）的训练样本，决定了 SVM 的决策边界 | 3 个样本中 A(3, 3)、C(1, 1) 是支持向量，B(4, 3) 不是 |
| <span id="g-硬间隔"></span>硬间隔 | Hard Margin | 要求所有训练样本都在间隔以外、一个都不能分错的 SVM | 数据中有一个录错的样本时，硬间隔把间隔压窄到 0.40 |
| <span id="g-软间隔"></span>软间隔 | Soft Margin | 允许少数样本进入间隔甚至被分错、但要为此付出代价的 SVM | C = 0.1 时放弃异常点，间隔保持 2.91 |
| <span id="g-合页损失"></span>合页损失 | Hinge Loss | SVM 使用的损失 max(0, 1 − y·f(x))：样本在间隔以外为 0，越界越深损失越大 | 正类样本 f(x) = 0.4 时损失 0.6，f(x) = −0.5 时损失 1.5 |
| <span id="g-惩罚参数-C"></span>惩罚参数 C | Penalty Parameter C | SVM 中合页损失的权重：C 大则尽量不让样本越界、间隔窄，C 小则允许越界、间隔宽 | C 从 0.01 调到 100，间隔从 4.04 缩到 0.68 |
| <span id="g-核函数"></span>核函数 | Kernel Function | 直接用原始特征算出两个样本在某个高维空间中的点积（相似程度）的函数 | (x·z)² 等于映射到“所有二次项”空间后的点积 |
| <span id="g-核技巧"></span>核技巧 | Kernel Trick | 用核函数代替高维空间中的点积，不必真的把数据映射到高维，就能得到弯曲的决策边界 | RBF 核 SVM 在同心圆数据上测试准确率 1.000 |
| <span id="g-线性核"></span>线性核 | Linear Kernel | K(x, z) = x·z，即不做映射，决策边界是直线（平面） | 同心圆数据上线性核测试准确率只有 0.489 |
| <span id="g-多项式核"></span>多项式核 | Polynomial Kernel | K(x, z) = (γ·x·z + r)^d，相当于加入 d 次以内的多项式特征 | 2 次、coef0 = 1 的多项式核能画出圆形边界 |
| <span id="g-RBF-核"></span>RBF 核 | RBF Kernel | K(x, z) = exp(−γ‖x − z‖²)，按两个样本的距离衡量相似程度，又叫高斯核 | γ = 1 时距离为 1 的两个样本相似度 0.368 |
| <span id="g-gamma-参数"></span>gamma 参数 | Gamma | RBF 核中控制每个样本影响范围的参数：gamma 越大影响范围越小，边界越曲折，越容易过拟合 | 月牙数据 gamma = 100 时训练准确率 1.000，交叉验证只有 0.79～0.91 |
| <span id="g-支持向量回归"></span>支持向量回归 | Support Vector Regression, SVR | 用 SVM 思路做回归：在预测值周围留一条宽 ε 的管道，管道内的误差不计，只惩罚管道外的样本 | 奶茶店销量 ε = 5 时测试 MSE 623.7 |
| <span id="g-一对一"></span>一对一 | One-vs-One | 多分类策略：每两类训练一个二分类器，K 类共 K(K−1)/2 个，预测时投票决定类别 | 10 个数字要训练 45 个 SVM |

## 第 7 章 K 近邻、朴素贝叶斯与关联规则

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-K-近邻"></span>K 近邻 | K-Nearest Neighbors, KNN | 找出与新样本最相近的 K 个已知样本，按它们的类别投票决定新样本类别的方法 | 葡萄酒的 5 个最近邻都是品种 0，判为品种 0 |
| <span id="g-距离度量"></span>距离度量 | Distance Metric | 衡量两个样本相差多远（或多相似）的计算方法，如欧氏距离、曼哈顿距离、余弦相似度 | KNN 中 `metric="euclidean"` 或 `"manhattan"` |
| <span id="g-欧氏距离"></span>欧氏距离 | Euclidean Distance | 两点之间的直线距离：各特征之差的平方和再开平方 | (0, 0) 与 (3, 4) 的欧氏距离为 5 |
| <span id="g-曼哈顿距离"></span>曼哈顿距离 | Manhattan Distance | 各特征之差的绝对值之和，像在棋盘式街道上只能横走竖走的距离 | (0, 0) 与 (3, 4) 的曼哈顿距离为 7 |
| <span id="g-余弦相似度"></span>余弦相似度 | Cosine Similarity | 两个向量夹角的余弦值，只看方向是否一致、不看长短，常用于文本和嵌入向量 | (1, 2) 与 (2, 4) 的余弦相似度为 1 |
| <span id="g-K-值选择"></span>K 值选择 | Choosing K | 确定 KNN 中邻居个数 K 的过程：K 太小易过拟合，太大易欠拟合，一般用交叉验证选 | 葡萄酒数据交叉验证选出 K = 12 |
| <span id="g-维度灾难"></span>维度灾难 | Curse of Dimensionality | 特征维数很高时，样本变得稀疏、彼此距离趋于相同，依赖距离的方法效果变差的现象 | 加入 1000 个随机特征后 KNN 准确率从 0.963 降到 0.537 |
| <span id="g-条件概率"></span>条件概率 | Conditional Probability | 在事件 B 已经发生的条件下事件 A 发生的概率，记作 P(A \| B) | 含“免费”的 20 条短信中 15 条是垃圾短信，P = 0.75 |
| <span id="g-贝叶斯定理"></span>贝叶斯定理 | Bayes' Theorem | 用 P(B\|A)、P(A) 和 P(B) 计算 P(A\|B) 的公式，把“原因导致结果的概率”反推为“看到结果后原因的概率” | 患病率 1% 时检测阳性者真正患病的概率约 16.1% |
| <span id="g-先验概率与后验概率"></span>先验概率与后验概率 | Prior and Posterior Probability | 看到证据之前对某事发生可能性的估计叫先验概率，看到证据后用贝叶斯定理更新得到的叫后验概率 | 患病的先验概率 1%，检测阳性后的后验概率 16.1% |
| <span id="g-朴素贝叶斯"></span>朴素贝叶斯 | Naive Bayes | 基于贝叶斯定理、假设各个词相互独立的分类算法 | 垃圾短信过滤测试准确率 0.984 |
| <span id="g-条件独立假设"></span>条件独立假设 | Conditional Independence Assumption | 朴素贝叶斯的假设：类别确定后，各个特征之间相互独立，于是整体的条件概率等于各特征条件概率的乘积 | P(免费 中奖 \| 垃圾) = P(免费 \| 垃圾) × P(中奖 \| 垃圾) |
| <span id="g-拉普拉斯平滑"></span>拉普拉斯平滑 | Laplace Smoothing | 计算条件概率时给每个词的出现次数都加 1（分母相应加上词表大小），避免没见过的词使概率变成 0 | “开会”在垃圾短信中出现 0 次，平滑后概率为 1/30 |
| <span id="g-高斯朴素贝叶斯"></span>高斯朴素贝叶斯 | Gaussian Naive Bayes | 假设每个连续特征在每一类中服从正态分布、按正态分布计算条件概率的朴素贝叶斯 | 用身高、血压等连续指标分类 |
| <span id="g-多项式朴素贝叶斯"></span>多项式朴素贝叶斯 | Multinomial Naive Bayes | 按每个词在某类中出现次数的比例计算条件概率的朴素贝叶斯，适合词袋等计数特征 | scikit-learn 的 `MultinomialNB` |
| <span id="g-关联规则"></span>关联规则 | Association Rule | 形如“如果买了 A，那么也常买 B”的规律 A → B，用支持度、置信度和提升度衡量 | {面包} → {牛奶} |
| <span id="g-项集"></span>项集 | Itemset | 一组商品（项）的集合，包含 k 件商品的叫 k 项集 | {面包, 牛奶} 是一个 2 项集 |
| <span id="g-支持度"></span>支持度 | Support | 同时包含某个项集的小票占全部小票的比例；规则 A → B 的支持度是同时含 A 和 B 的比例 | 10 张小票中 4 张同时有面包和牛奶，支持度 0.4 |
| <span id="g-频繁项集"></span>频繁项集 | Frequent Itemset | 支持度不低于最小支持度的项集 | 最小支持度 0.02 时 3000 张小票中有 80 个频繁项集 |
| <span id="g-置信度"></span>置信度 | Confidence | 规则 A → B 中，含 A 的小票里同时含 B 的比例，即条件概率 P(B \| A) | 买面包的 5 张小票中 4 张也有牛奶，置信度 0.8 |
| <span id="g-提升度"></span>提升度 | Lift | 置信度除以后件的支持度：大于 1 表示买了 A 的人更可能买 B，等于 1 表示无关，小于 1 表示负相关 | {火锅底料} → {肥牛卷} 提升度 11.0 |
| <span id="g-Apriori-算法"></span>Apriori 算法 | Apriori Algorithm | 利用“频繁项集的子集也一定频繁”的性质，从 1 项集开始逐层生成、筛选候选项集的频繁项集挖掘算法 | 花生不频繁，所有含花生的组合直接跳过 |
| <span id="g-FP-Growth-算法"></span>FP-Growth 算法 | FP-Growth Algorithm | 先把所有小票压缩成一棵频繁模式树，再在树上直接挖掘频繁项集、不生成候选项集的算法 | mlxtend 的 `fpgrowth`，结果与 `apriori` 相同 |
| <span id="g-购物篮分析"></span>购物篮分析 | Market Basket Analysis | 从大量购物记录中找出经常被一起购买的商品组合，用于摆货、促销和推荐 | 分析 3000 张社区超市小票 |
| <span id="g-推荐系统"></span>推荐系统 | Recommender System | 根据用户的行为和商品的特点，向用户推荐可能感兴趣的商品或内容的系统 | 放入“火锅底料”后推荐肥牛卷、丸子 |

## 第 8 章 聚类、降维与异常检测

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-聚类"></span>聚类 | Clustering | 把没有标签的样本按相似程度自动分成若干组的无监督学习任务 | 按 R、F、M 把 2000 位客户分成四群 |
| <span id="g-K-均值"></span>K 均值 | K-Means | 先选 K 个中心、再交替执行“分组”和“更新中心”的聚类方法 | 6 个点、初始中心 A 和 B，迭代两轮后收敛到 (1.33, 1.33) 和 (6.33, 5.33) |
| <span id="g-簇中心"></span>簇中心 | Cluster Center, Centroid | K 均值中代表一个簇的点，等于簇内所有样本的平均位置 | `KMeans` 的 `cluster_centers_` |
| <span id="g-肘部法则"></span>肘部法则 | Elbow Method | 画出簇内平方和随 K 变化的曲线，取下降由快变慢的拐点作为 K 的方法 | 实验 8-1 中曲线在 K = 4 之后变缓 |
| <span id="g-轮廓系数"></span>轮廓系数 | Silhouette Coefficient | 用 s = (b − a) / max(a, b) 衡量样本与本簇是否紧凑、与别簇是否分开的指标，取值 −1 到 1，越大越好 | 实验 8-1 中 K = 4 时最高，为 0.433 |
| <span id="g-层次聚类"></span>层次聚类 | Hierarchical Clustering | 从每个样本自成一簇开始，每次合并最相近的两个簇，逐层形成树状结构的聚类方法 | 单链接的层次聚类完全分开两个月牙 |
| <span id="g-树状图"></span>树状图 | Dendrogram | 画出层次聚类合并过程的树形图，纵轴是合并时的距离，横切一刀得到若干簇 | 在高度 3 处切，得到 3 个簇 |
| <span id="g-DBSCAN"></span>DBSCAN | Density-Based Spatial Clustering of Applications with Noise | 按密度找簇的聚类方法：密集区域连成簇，稀疏处的点作为噪声点，不需要事先给出簇数 | `DBSCAN(eps=0.3, min_samples=5)` |
| <span id="g-密度可达"></span>密度可达 | Density-Reachable | DBSCAN 中，从一个核心点出发，经过一串相邻的核心点能够到达某个点的关系；相互密度可达的点属于同一簇 | 一维数据 1、2、3、4 中，1 和 4 经核心点 2、3 相互密度可达 |
| <span id="g-噪声点"></span>噪声点 | Noise Point | DBSCAN 中既不是核心点、也不在任何核心点邻域内的点，不属于任何簇，编号为 −1 | 1、2、3、4、10、11、12、20 中的 20 |
| <span id="g-客户分群"></span>客户分群 | Customer Segmentation | 按消费行为等特征把客户分成若干群、对不同群采取不同运营策略的做法，常用 RFM 模型 | 高价值忠实、流失风险、新客与潜力、低价值沉睡 |
| <span id="g-降维"></span>降维 | Dimensionality Reduction | 把很多个特征压缩成少数几个新特征，同时尽量保留原有信息的方法 | 64 个像素特征降到 2 维画图 |
| <span id="g-主成分分析"></span>主成分分析 | Principal Component Analysis, PCA | 寻找数据方差最大的若干个相互垂直的方向，把数据投影到这些方向上实现降维的方法 | 购买次数和消费金额压成一个“购买力”分数，保留 98.5% 的方差 |
| <span id="g-方差解释率"></span>方差解释率 | Explained Variance Ratio | 一个主成分保留的方差占全部方差的比例，用于决定保留几个主成分 | 手写数字前 2 个主成分合计 28.5% |
| <span id="g-t-SNE"></span>t-SNE | t-Distributed Stochastic Neighbor Embedding | 让高维空间中相近的样本在二维图上也相近的非线性降维方法，只用于可视化 | 手写数字的 10 个数字在 t-SNE 图上各成一团 |
| <span id="g-异常检测"></span>异常检测 | Anomaly Detection | 从大量数据中找出与大多数样本明显不同的少数样本的任务，常用无监督方法 | 在 20000 笔交易中找出可疑交易 |
| <span id="g-孤立森林"></span>孤立森林 | Isolation Forest | 用许多棵随机切分的树，按样本被单独切出所需的平均次数衡量异常程度的方法，次数越少越异常 | `IsolationForest(contamination=0.01)` 标出 200 笔，其中 54 笔是欺诈 |

## 第 9 章 模型选择、调优与部署

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-偏差与方差"></span>偏差与方差 | Bias and Variance | 模型误差的两个来源：偏差是模型假设与真实规律的系统性差距（欠拟合），方差是模型对训练数据变化的敏感程度（过拟合） | 线性回归偏差大；不限深度的决策树方差大 |
| <span id="g-交叉验证"></span>交叉验证 | Cross Validation | 把数据分成几份、轮流做测试集，取多次成绩平均值的评价方法 | 深度 5 的决策树 5 折交叉验证平均 0.780 |
| <span id="g-K-折交叉验证"></span>K 折交叉验证 | K-Fold Cross Validation | 把数据平均分成 K 份，做 K 轮，每轮用一份验证、其余 K − 1 份训练，取 K 个分数平均的方法 | `cross_val_score(model, X, y, cv=5)` |
| <span id="g-分层抽样"></span>分层抽样 | Stratified Sampling | 按类别比例抽取样本，使每一份中各类别的比例与整体一致的抽样方法 | `StratifiedKFold`、`train_test_split(..., stratify=y)` |
| <span id="g-时间序列交叉验证"></span>时间序列交叉验证 | Time Series Cross Validation | 按时间顺序划分、每一折只用过去的数据训练并预测之后一段的交叉验证方法 | 用 1～4 月训练、预测 5～6 月 |
| <span id="g-学习曲线"></span>学习曲线 | Learning Curve | 训练得分和验证得分随训练样本数变化的曲线，用于诊断欠拟合和过拟合 | 不限深度决策树训练 1.000、验证 0.707 |
| <span id="g-验证曲线"></span>验证曲线 | Validation Curve | 训练得分和验证得分随某个超参数取值变化的曲线，用于选择超参数 | 决策树最大深度 4 时验证得分最高 |
| <span id="g-网格搜索"></span>网格搜索 | Grid Search | 列出每个超参数的候选值，试遍所有组合并用交叉验证打分的调参方法 | 3 × 4 × 3 = 36 种组合 |
| <span id="g-随机搜索"></span>随机搜索 | Random Search | 在给定范围内随机抽取若干组超参数进行试验的调参方法 | `RandomizedSearchCV(n_iter=36)` |
| <span id="g-Optuna-调参"></span>Optuna 调参 | Optuna Tuning | 用 Optuna 库根据已有试验结果决定下一次试验取值的自动调参方法 | 36 次试验，交叉验证 AUC 0.842 → 0.865 |
| <span id="g-流水线"></span>流水线 | Pipeline | 把预处理步骤和模型按顺序串成一个整体，统一 fit 和 predict，防止数据泄露 | 列转换器 + 逻辑回归 |
| <span id="g-列转换器"></span>列转换器 | ColumnTransformer | 对表格中不同的列分别做不同预处理再拼接结果的工具 | 数值列标准化，类别列独热编码 |
| <span id="g-置换重要性"></span>置换重要性 | Permutation Importance | 把某一列的值随机打乱后模型成绩下降的幅度，用来衡量模型对该特征的依赖程度 | 打乱合约类型，测试 AUC 下降 0.140 |
| <span id="g-SHAP-值"></span>SHAP 值 | SHAP Value | 把一次预测分解为基准值加各特征贡献的解释方法，贡献为正推高预测、为负拉低预测 | 在网 2 个月使流失的对数几率增加 1.402 |
| <span id="g-模型保存"></span>模型保存 | Model Persistence | 把训练好的模型存成文件，以后直接加载使用而不必重新训练 | 保存整条流水线，69 KB |
| <span id="g-joblib"></span>joblib | joblib | scikit-learn 推荐的模型保存与加载工具 | `joblib.dump(pipe, "churn_model.joblib")` |
| <span id="g-模型接口"></span>模型接口 | Model API | 把模型放在服务器上，其他程序通过网络发送数据、获取预测结果的服务方式 | `POST /predict` 返回流失概率和风险等级 |
| <span id="g-FastAPI"></span>FastAPI | FastAPI | 用 Python 编写网络接口的框架，能自动检查请求数据并生成接口文档 | 不合法的请求返回状态码 422 |
| <span id="g-Gradio"></span>Gradio | Gradio | 用几行 Python 代码把函数变成可交互网页的工具，常用于模型演示 | `gr.Interface(fn=predict, ...)` |
| <span id="g-模型部署"></span>模型部署 | Model Deployment | 把训练好的模型交给使用者、在实际业务中运行的全部工作 | 用接口或网页提供流失预测服务 |
| <span id="g-数据漂移"></span>数据漂移 | Data Drift | 上线后输入数据的分布与训练时相比发生变化的现象，常用 PSI 监控 | 涨价后月消费的 PSI 为 0.313 |

## 第 10 章 神经网络基础

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-神经网络"></span>神经网络 | Neural Network | 由大量神经元按层连接而成、通过数据学习连接权重的模型 | 实验 10-2 的“64 → 64 → 10”网络 |
| <span id="g-神经元"></span>神经元 | Neuron | 神经网络的基本计算单元：对输入加权求和、加偏置，再经过激活函数得到输出 | 0.5 × 1 + 0.4 × 0 + 0.3 × 1 − 0.6 = 0.2，Sigmoid 后约 0.550 |
| <span id="g-感知机"></span>感知机 | Perceptron | 只有一个神经元、用阶跃函数输出 0 或 1 的分类模型，预测错时按 w ← w + η(y − ŷ)x 更新权重 | 学“与”5 轮后全对：w = [0.2, 0.1]，b = −0.2 |
| <span id="g-异或问题"></span>异或问题 | XOR Problem | 两个输入不同时输出 1、相同时输出 0，两类点交叉摆放、线性不可分，单层感知机无法解决 | (0, 1)、(1, 0) 为 1，(0, 0)、(1, 1) 为 0 |
| <span id="g-多层感知机"></span>多层感知机 | Multilayer Perceptron (MLP) | 由输入层、一个或多个隐藏层和输出层组成、相邻层全连接的前馈神经网络 | “2 → 4 → 1”的网络学会了异或 |
| <span id="g-激活函数"></span>激活函数 | Activation Function | 神经元对加权和做的非线性变换，使网络能表达非线性关系 | Sigmoid、Tanh、ReLU、Softmax |
| <span id="g-ReLU"></span>ReLU | Rectified Linear Unit | 修正线性单元，小于 0 输出 0、大于 0 原样输出的激活函数，隐藏层的首选 | ReLU(−2) = 0，ReLU(3) = 3 |
| <span id="g-隐藏层"></span>隐藏层 | Hidden Layer | 位于输入层和输出层之间的层，把输入变换成对任务更有用的新特征 | 异或网络的 4 个隐藏神经元 |
| <span id="g-前向传播"></span>前向传播 | Forward Propagation | 输入数据从输入层出发逐层计算，直到得到输出和损失的过程 | z1 = X @ W1 + b1，a1 = sigmoid(z1)，…… |
| <span id="g-反向传播"></span>反向传播 | Backpropagation | 从损失出发、利用链式法则逐层往回计算每个参数梯度的算法 | 手算例子中 dL/dw₂ = 2，dL/dw₁ = 12 |
| <span id="g-链式法则"></span>链式法则 | Chain Rule | 复合函数求导的法则：L 通过 y 依赖于 x 时，dL/dx = dL/dy × dy/dx | (3w − 4)² 在 w = 2 处的导数 2 × 2 × 3 = 12 |
| <span id="g-梯度消失"></span>梯度消失 | Vanishing Gradient | 误差往回传时层层乘上小于 1 的导数，使前面层的梯度趋近于 0、几乎无法更新的现象 | 10 层 Sigmoid 网络第 1 层的梯度是输出层的 1.5 × 10⁻⁹ |
| <span id="g-优化器"></span>优化器 | Optimizer | 根据梯度决定怎样更新参数的算法 | SGD、带动量的 SGD、Adam |
| <span id="g-Adam"></span>Adam | Adam | 结合动量并为每个参数自适应调整步长的优化器，默认学习率 0.001 | 学习率 0.01 时第 3 轮验证准确率就达到 0.95 |
| <span id="g-轮次与批大小"></span>轮次与批大小 | Epoch and Batch Size | 轮次是完整看一遍全部训练数据的次数，批大小是每次更新参数所用的样本数 | 1005 个样本、批大小 32，每轮更新 32 次 |
| <span id="g-Dropout"></span>Dropout | Dropout | 训练时每一步随机让一部分神经元输出为 0 以减轻过拟合的方法，预测时不丢弃 | `nn.Dropout(0.5)`、`keras.layers.Dropout(0.2)` |
| <span id="g-批归一化"></span>批归一化 | Batch Normalization | 在网络层中用当前批数据的均值和标准差做标准化，再做可学习的缩放和平移，使深层网络训练更快更稳 | 4 层网络加批归一化后测试准确率从 0.113 升到 0.972 |
| <span id="g-PyTorch"></span>PyTorch | PyTorch | Meta 开源的深度学习框架，以张量和自动求导为核心 | `nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))` |
| <span id="g-张量"></span>张量 | Tensor | 深度学习框架中存放数据的多维数组，可以放在 GPU 上计算并支持自动求导 | 一批彩色图片：张数 × 通道 × 高 × 宽 |
| <span id="g-自动求导"></span>自动求导 | Automatic Differentiation | 框架记录计算过程并自动用反向传播求出梯度的功能 | `loss.backward()` 后梯度存在 `.grad` 中 |
| <span id="g-Keras"></span>Keras | Keras | 高层深度学习接口，用 Sequential 等方式搭建网络，Keras 3 可选 TensorFlow、PyTorch、JAX 作后端 | `model.compile(...)`、`model.fit(...)` |
| <span id="g-TensorFlow"></span>TensorFlow | TensorFlow | 谷歌开源的深度学习框架，在移动端和生产部署方面工具丰富 | Keras 3 的默认后端 |

## 第 11 章 卷积神经网络与循环神经网络

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|

## 第 12 章 时间序列预测与强化学习入门

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
