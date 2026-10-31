---
title: 学习图谱
description: 全书核心概念及其先修关系，点一个概念就能看到学习路径
---

# 学习图谱

学习图谱把全书的核心概念和它们之间的“先修关系”画在一张图上：要理解一个概念，得先懂哪些更基础的概念；顺着箭头往回找，就是一条学习路径。

<iframe class="sim-frame" src="main.html" height="900" loading="lazy" title="学习图谱"></iframe>

[全屏打开](main.html){ target=_blank }

## 怎样使用

- **预习**：学新的一章前，点开本章的概念，看看橙色的先修概念自己是否都掌握了，没掌握的先回去复习；
- **复习**：期末复习时，从最后几章的概念出发，顺着学习路径往回梳理，就能把全书串起来；
- **查漏补缺**：做练习时遇到不懂的概念，在搜索框里输入它，看它依赖哪些更基础的概念。

<!-- graph:start -->

图谱共有 **283 个概念、380 条先修关系**，概念按章排列，颜色表示 10 个类别（基础概念、数据与特征、回归、分类、树模型与集成、评估与调优、无监督学习、神经网络、序列与时间序列、强化学习与责任）。概念的定义都来自[附录 D 术语表](../appendix/glossary.md)，先修关系由编者根据各章内容整理。

## 文字版

下面按章列出每个概念的直接先修概念，便于打印或用读屏软件阅读。

??? note "第 1 章 概述与流程（24 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [机器学习](../appendix/glossary.md#g-机器学习) | 基础概念 | — |
    | [深度学习](../appendix/glossary.md#g-深度学习) | 基础概念 | [机器学习](../appendix/glossary.md#g-机器学习) |
    | [样本](../appendix/glossary.md#g-样本) | 基础概念 | [机器学习](../appendix/glossary.md#g-机器学习) |
    | [特征](../appendix/glossary.md#g-特征) | 基础概念 | [样本](../appendix/glossary.md#g-样本) |
    | [标签](../appendix/glossary.md#g-标签) | 基础概念 | [样本](../appendix/glossary.md#g-样本) |
    | [数据集](../appendix/glossary.md#g-数据集) | 基础概念 | [样本](../appendix/glossary.md#g-样本) |
    | [模型](../appendix/glossary.md#g-模型) | 基础概念 | [机器学习](../appendix/glossary.md#g-机器学习) |
    | [模型参数](../appendix/glossary.md#g-模型参数) | 基础概念 | [模型](../appendix/glossary.md#g-模型) |
    | [训练](../appendix/glossary.md#g-训练) | 基础概念 | [模型](../appendix/glossary.md#g-模型)、[数据集](../appendix/glossary.md#g-数据集) |
    | [预测](../appendix/glossary.md#g-预测) | 基础概念 | [模型](../appendix/glossary.md#g-模型) |
    | [泛化能力](../appendix/glossary.md#g-泛化能力) | 基础概念 | [训练](../appendix/glossary.md#g-训练)、[预测](../appendix/glossary.md#g-预测) |
    | [训练集](../appendix/glossary.md#g-训练集) | 基础概念 | [数据集](../appendix/glossary.md#g-数据集) |
    | [测试集](../appendix/glossary.md#g-测试集) | 基础概念 | [数据集](../appendix/glossary.md#g-数据集) |
    | [验证集](../appendix/glossary.md#g-验证集) | 评估与调优 | [训练集](../appendix/glossary.md#g-训练集)、[测试集](../appendix/glossary.md#g-测试集) |
    | [监督学习](../appendix/glossary.md#g-监督学习) | 基础概念 | [标签](../appendix/glossary.md#g-标签)、[训练](../appendix/glossary.md#g-训练) |
    | [回归](../appendix/glossary.md#g-回归) | 基础概念 | [监督学习](../appendix/glossary.md#g-监督学习) |
    | [分类](../appendix/glossary.md#g-分类) | 基础概念 | [监督学习](../appendix/glossary.md#g-监督学习) |
    | [无监督学习](../appendix/glossary.md#g-无监督学习) | 基础概念 | [机器学习](../appendix/glossary.md#g-机器学习)、[特征](../appendix/glossary.md#g-特征) |
    | [强化学习](../appendix/glossary.md#g-强化学习) | 基础概念 | [机器学习](../appendix/glossary.md#g-机器学习) |
    | [半监督学习](../appendix/glossary.md#g-半监督学习) | 基础概念 | [监督学习](../appendix/glossary.md#g-监督学习)、[无监督学习](../appendix/glossary.md#g-无监督学习) |
    | [自监督学习](../appendix/glossary.md#g-自监督学习) | 基础概念 | [无监督学习](../appendix/glossary.md#g-无监督学习) |
    | [机器学习项目流程](../appendix/glossary.md#g-机器学习项目流程) | 基础概念 | [机器学习](../appendix/glossary.md#g-机器学习) |
    | [scikit-learn](../appendix/glossary.md#g-scikit-learn) | 基础概念 | [机器学习](../appendix/glossary.md#g-机器学习) |
    | [超参数](../appendix/glossary.md#g-超参数) | 评估与调优 | [模型](../appendix/glossary.md#g-模型) |

??? note "第 2 章 预处理与特征工程（31 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [数据质量](../appendix/glossary.md#g-数据质量) | 数据与特征 | [数据集](../appendix/glossary.md#g-数据集) |
    | [探索性数据分析](../appendix/glossary.md#g-探索性数据分析) | 数据与特征 | [数据质量](../appendix/glossary.md#g-数据质量) |
    | [描述性统计](../appendix/glossary.md#g-描述性统计) | 数据与特征 | [探索性数据分析](../appendix/glossary.md#g-探索性数据分析) |
    | [数据透视](../appendix/glossary.md#g-数据透视) | 数据与特征 | [探索性数据分析](../appendix/glossary.md#g-探索性数据分析) |
    | [相关系数](../appendix/glossary.md#g-相关系数) | 数据与特征 | [描述性统计](../appendix/glossary.md#g-描述性统计) |
    | [数据清洗](../appendix/glossary.md#g-数据清洗) | 数据与特征 | [数据质量](../appendix/glossary.md#g-数据质量) |
    | [缺失值](../appendix/glossary.md#g-缺失值) | 数据与特征 | [数据质量](../appendix/glossary.md#g-数据质量)、[描述性统计](../appendix/glossary.md#g-描述性统计) |
    | [异常值](../appendix/glossary.md#g-异常值) | 数据与特征 | [数据质量](../appendix/glossary.md#g-数据质量)、[描述性统计](../appendix/glossary.md#g-描述性统计) |
    | [3σ 法则](../appendix/glossary.md#g-3σ-法则) | 数据与特征 | [异常值](../appendix/glossary.md#g-异常值) |
    | [箱线图法](../appendix/glossary.md#g-箱线图法) | 数据与特征 | [异常值](../appendix/glossary.md#g-异常值)、[描述性统计](../appendix/glossary.md#g-描述性统计) |
    | [重复值](../appendix/glossary.md#g-重复值) | 数据与特征 | [数据质量](../appendix/glossary.md#g-数据质量) |
    | [数据清洗规范](../appendix/glossary.md#g-数据清洗规范) | 数据与特征 | [缺失值](../appendix/glossary.md#g-缺失值)、[异常值](../appendix/glossary.md#g-异常值)、[重复值](../appendix/glossary.md#g-重复值) |
    | [数值型特征](../appendix/glossary.md#g-数值型特征) | 数据与特征 | [特征](../appendix/glossary.md#g-特征) |
    | [类别型特征](../appendix/glossary.md#g-类别型特征) | 数据与特征 | [特征](../appendix/glossary.md#g-特征) |
    | [标签编码](../appendix/glossary.md#g-标签编码) | 数据与特征 | [类别型特征](../appendix/glossary.md#g-类别型特征) |
    | [独热编码](../appendix/glossary.md#g-独热编码) | 数据与特征 | [类别型特征](../appendix/glossary.md#g-类别型特征) |
    | [高基数类别特征](../appendix/glossary.md#g-高基数类别特征) | 数据与特征 | [类别型特征](../appendix/glossary.md#g-类别型特征) |
    | [目标编码](../appendix/glossary.md#g-目标编码) | 数据与特征 | [高基数类别特征](../appendix/glossary.md#g-高基数类别特征)、[数据泄露](../appendix/glossary.md#g-数据泄露) |
    | [特征缩放](../appendix/glossary.md#g-特征缩放) | 数据与特征 | [数值型特征](../appendix/glossary.md#g-数值型特征) |
    | [标准化](../appendix/glossary.md#g-标准化) | 数据与特征 | [数值型特征](../appendix/glossary.md#g-数值型特征)、[描述性统计](../appendix/glossary.md#g-描述性统计)、[特征缩放](../appendix/glossary.md#g-特征缩放) |
    | [归一化](../appendix/glossary.md#g-归一化) | 数据与特征 | [数值型特征](../appendix/glossary.md#g-数值型特征)、[特征缩放](../appendix/glossary.md#g-特征缩放) |
    | [特征工程](../appendix/glossary.md#g-特征工程) | 数据与特征 | [特征](../appendix/glossary.md#g-特征) |
    | [特征构造](../appendix/glossary.md#g-特征构造) | 数据与特征 | [特征](../appendix/glossary.md#g-特征) |
    | [日期特征](../appendix/glossary.md#g-日期特征) | 数据与特征 | [特征构造](../appendix/glossary.md#g-特征构造) |
    | [交叉特征](../appendix/glossary.md#g-交叉特征) | 数据与特征 | [特征构造](../appendix/glossary.md#g-特征构造) |
    | [分箱](../appendix/glossary.md#g-分箱) | 数据与特征 | [数值型特征](../appendix/glossary.md#g-数值型特征) |
    | [特征选择](../appendix/glossary.md#g-特征选择) | 数据与特征 | [特征](../appendix/glossary.md#g-特征) |
    | [过滤法](../appendix/glossary.md#g-过滤法) | 数据与特征 | [特征选择](../appendix/glossary.md#g-特征选择)、[相关系数](../appendix/glossary.md#g-相关系数) |
    | [包裹法](../appendix/glossary.md#g-包裹法) | 数据与特征 | [特征选择](../appendix/glossary.md#g-特征选择) |
    | [嵌入法](../appendix/glossary.md#g-嵌入法) | 数据与特征 | [特征选择](../appendix/glossary.md#g-特征选择) |
    | [数据泄露](../appendix/glossary.md#g-数据泄露) | 数据与特征 | [训练集](../appendix/glossary.md#g-训练集)、[测试集](../appendix/glossary.md#g-测试集)、[特征构造](../appendix/glossary.md#g-特征构造) |

??? note "第 3 章 线性回归（24 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [线性回归](../appendix/glossary.md#g-线性回归) | 回归 | [回归](../appendix/glossary.md#g-回归)、[数值型特征](../appendix/glossary.md#g-数值型特征) |
    | [一元线性回归](../appendix/glossary.md#g-一元线性回归) | 回归 | [线性回归](../appendix/glossary.md#g-线性回归) |
    | [多元线性回归](../appendix/glossary.md#g-多元线性回归) | 回归 | [一元线性回归](../appendix/glossary.md#g-一元线性回归) |
    | [权重与偏置](../appendix/glossary.md#g-权重与偏置) | 回归 | [线性回归](../appendix/glossary.md#g-线性回归) |
    | [损失函数](../appendix/glossary.md#g-损失函数) | 回归 | [模型](../appendix/glossary.md#g-模型) |
    | [残差](../appendix/glossary.md#g-残差) | 回归 | [线性回归](../appendix/glossary.md#g-线性回归) |
    | [均方误差](../appendix/glossary.md#g-均方误差) | 回归 | [损失函数](../appendix/glossary.md#g-损失函数)、[残差](../appendix/glossary.md#g-残差) |
    | [最小二乘法](../appendix/glossary.md#g-最小二乘法) | 回归 | [均方误差](../appendix/glossary.md#g-均方误差)、[一元线性回归](../appendix/glossary.md#g-一元线性回归) |
    | [梯度](../appendix/glossary.md#g-梯度) | 回归 | [损失函数](../appendix/glossary.md#g-损失函数) |
    | [梯度下降](../appendix/glossary.md#g-梯度下降) | 回归 | [梯度](../appendix/glossary.md#g-梯度) |
    | [学习率](../appendix/glossary.md#g-学习率) | 回归 | [梯度下降](../appendix/glossary.md#g-梯度下降) |
    | [随机梯度下降](../appendix/glossary.md#g-随机梯度下降) | 回归 | [梯度下降](../appendix/glossary.md#g-梯度下降) |
    | [小批量梯度下降](../appendix/glossary.md#g-小批量梯度下降) | 回归 | [随机梯度下降](../appendix/glossary.md#g-随机梯度下降) |
    | [多重共线性](../appendix/glossary.md#g-多重共线性) | 回归 | [多元线性回归](../appendix/glossary.md#g-多元线性回归)、[相关系数](../appendix/glossary.md#g-相关系数) |
    | [平均绝对误差](../appendix/glossary.md#g-平均绝对误差) | 回归 | [回归](../appendix/glossary.md#g-回归) |
    | [均方根误差](../appendix/glossary.md#g-均方根误差) | 回归 | [均方误差](../appendix/glossary.md#g-均方误差) |
    | [决定系数](../appendix/glossary.md#g-决定系数) | 回归 | [均方误差](../appendix/glossary.md#g-均方误差) |
    | [残差图](../appendix/glossary.md#g-残差图) | 回归 | [残差](../appendix/glossary.md#g-残差) |
    | [多项式回归](../appendix/glossary.md#g-多项式回归) | 回归 | [多元线性回归](../appendix/glossary.md#g-多元线性回归)、[特征构造](../appendix/glossary.md#g-特征构造) |
    | [欠拟合](../appendix/glossary.md#g-欠拟合) | 回归 | [泛化能力](../appendix/glossary.md#g-泛化能力) |
    | [过拟合](../appendix/glossary.md#g-过拟合) | 回归 | [泛化能力](../appendix/glossary.md#g-泛化能力)、[多项式回归](../appendix/glossary.md#g-多项式回归) |
    | [正则化](../appendix/glossary.md#g-正则化) | 回归 | [过拟合](../appendix/glossary.md#g-过拟合) |
    | [岭回归](../appendix/glossary.md#g-岭回归) | 回归 | [正则化](../appendix/glossary.md#g-正则化) |
    | [Lasso 回归](../appendix/glossary.md#g-Lasso-回归) | 回归 | [正则化](../appendix/glossary.md#g-正则化) |

??? note "第 4 章 逻辑回归与评估（26 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [二分类](../appendix/glossary.md#g-二分类) | 分类 | [分类](../appendix/glossary.md#g-分类) |
    | [正类](../appendix/glossary.md#g-正类) | 分类 | [二分类](../appendix/glossary.md#g-二分类) |
    | [多分类](../appendix/glossary.md#g-多分类) | 分类 | [二分类](../appendix/glossary.md#g-二分类) |
    | [逻辑回归](../appendix/glossary.md#g-逻辑回归) | 分类 | [线性回归](../appendix/glossary.md#g-线性回归)、[分类](../appendix/glossary.md#g-分类) |
    | [Sigmoid 函数](../appendix/glossary.md#g-Sigmoid-函数) | 分类 | [逻辑回归](../appendix/glossary.md#g-逻辑回归) |
    | [概率输出](../appendix/glossary.md#g-概率输出) | 分类 | [Sigmoid 函数](../appendix/glossary.md#g-Sigmoid-函数) |
    | [分类阈值](../appendix/glossary.md#g-分类阈值) | 分类 | [概率输出](../appendix/glossary.md#g-概率输出) |
    | [决策边界](../appendix/glossary.md#g-决策边界) | 分类 | [逻辑回归](../appendix/glossary.md#g-逻辑回归) |
    | [交叉熵](../appendix/glossary.md#g-交叉熵) | 分类 | [逻辑回归](../appendix/glossary.md#g-逻辑回归)、[损失函数](../appendix/glossary.md#g-损失函数) |
    | [几率](../appendix/glossary.md#g-几率) | 分类 | [概率输出](../appendix/glossary.md#g-概率输出) |
    | [Softmax](../appendix/glossary.md#g-Softmax) | 分类 | [多分类](../appendix/glossary.md#g-多分类)、[Sigmoid 函数](../appendix/glossary.md#g-Sigmoid-函数) |
    | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) | 评估与调优 | [二分类](../appendix/glossary.md#g-二分类) |
    | [准确率](../appendix/glossary.md#g-准确率) | 评估与调优 | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) |
    | [精确率](../appendix/glossary.md#g-精确率) | 评估与调优 | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) |
    | [召回率](../appendix/glossary.md#g-召回率) | 评估与调优 | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) |
    | [F1 值](../appendix/glossary.md#g-F1-值) | 评估与调优 | [精确率](../appendix/glossary.md#g-精确率)、[召回率](../appendix/glossary.md#g-召回率) |
    | [假阳性率](../appendix/glossary.md#g-假阳性率) | 评估与调优 | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) |
    | [ROC 曲线](../appendix/glossary.md#g-ROC-曲线) | 评估与调优 | [分类阈值](../appendix/glossary.md#g-分类阈值)、[召回率](../appendix/glossary.md#g-召回率)、[假阳性率](../appendix/glossary.md#g-假阳性率) |
    | [AUC](../appendix/glossary.md#g-AUC) | 评估与调优 | [ROC 曲线](../appendix/glossary.md#g-ROC-曲线) |
    | [PR 曲线](../appendix/glossary.md#g-PR-曲线) | 评估与调优 | [精确率](../appendix/glossary.md#g-精确率)、[召回率](../appendix/glossary.md#g-召回率)、[分类阈值](../appendix/glossary.md#g-分类阈值) |
    | [类别不平衡](../appendix/glossary.md#g-类别不平衡) | 评估与调优 | [二分类](../appendix/glossary.md#g-二分类)、[准确率](../appendix/glossary.md#g-准确率) |
    | [类别权重](../appendix/glossary.md#g-类别权重) | 评估与调优 | [类别不平衡](../appendix/glossary.md#g-类别不平衡) |
    | [过采样与欠采样](../appendix/glossary.md#g-过采样与欠采样) | 评估与调优 | [类别不平衡](../appendix/glossary.md#g-类别不平衡) |
    | [SMOTE](../appendix/glossary.md#g-SMOTE) | 评估与调优 | [过采样与欠采样](../appendix/glossary.md#g-过采样与欠采样) |
    | [错误案例分析](../appendix/glossary.md#g-错误案例分析) | 评估与调优 | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) |
    | [模型测试报告](../appendix/glossary.md#g-模型测试报告) | 评估与调优 | [错误案例分析](../appendix/glossary.md#g-错误案例分析)、[F1 值](../appendix/glossary.md#g-F1-值) |

??? note "第 5 章 决策树与集成（28 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [决策树](../appendix/glossary.md#g-决策树) | 树模型与集成 | [分类](../appendix/glossary.md#g-分类)、[特征](../appendix/glossary.md#g-特征) |
    | [根节点](../appendix/glossary.md#g-根节点) | 树模型与集成 | [决策树](../appendix/glossary.md#g-决策树) |
    | [叶节点](../appendix/glossary.md#g-叶节点) | 树模型与集成 | [决策树](../appendix/glossary.md#g-决策树)、[根节点](../appendix/glossary.md#g-根节点) |
    | [树的深度](../appendix/glossary.md#g-树的深度) | 树模型与集成 | [决策树](../appendix/glossary.md#g-决策树) |
    | [信息熵](../appendix/glossary.md#g-信息熵) | 树模型与集成 | [决策树](../appendix/glossary.md#g-决策树) |
    | [信息增益](../appendix/glossary.md#g-信息增益) | 树模型与集成 | [信息熵](../appendix/glossary.md#g-信息熵) |
    | [ID3 算法](../appendix/glossary.md#g-ID3-算法) | 树模型与集成 | [信息增益](../appendix/glossary.md#g-信息增益) |
    | [信息增益率](../appendix/glossary.md#g-信息增益率) | 树模型与集成 | [信息增益](../appendix/glossary.md#g-信息增益) |
    | [C4.5 算法](../appendix/glossary.md#g-C4.5-算法) | 树模型与集成 | [信息增益率](../appendix/glossary.md#g-信息增益率)、[ID3 算法](../appendix/glossary.md#g-ID3-算法) |
    | [基尼系数](../appendix/glossary.md#g-基尼系数) | 树模型与集成 | [决策树](../appendix/glossary.md#g-决策树) |
    | [CART 算法](../appendix/glossary.md#g-CART-算法) | 树模型与集成 | [基尼系数](../appendix/glossary.md#g-基尼系数) |
    | [回归树](../appendix/glossary.md#g-回归树) | 树模型与集成 | [CART 算法](../appendix/glossary.md#g-CART-算法)、[回归](../appendix/glossary.md#g-回归) |
    | [剪枝](../appendix/glossary.md#g-剪枝) | 树模型与集成 | [决策树](../appendix/glossary.md#g-决策树)、[过拟合](../appendix/glossary.md#g-过拟合) |
    | [预剪枝](../appendix/glossary.md#g-预剪枝) | 树模型与集成 | [剪枝](../appendix/glossary.md#g-剪枝) |
    | [后剪枝](../appendix/glossary.md#g-后剪枝) | 树模型与集成 | [剪枝](../appendix/glossary.md#g-剪枝) |
    | [集成学习](../appendix/glossary.md#g-集成学习) | 树模型与集成 | [决策树](../appendix/glossary.md#g-决策树) |
    | [Bagging](../appendix/glossary.md#g-Bagging) | 树模型与集成 | [集成学习](../appendix/glossary.md#g-集成学习) |
    | [Boosting](../appendix/glossary.md#g-Boosting) | 树模型与集成 | [集成学习](../appendix/glossary.md#g-集成学习)、[弱学习器](../appendix/glossary.md#g-弱学习器) |
    | [自助采样](../appendix/glossary.md#g-自助采样) | 树模型与集成 | [Bagging](../appendix/glossary.md#g-Bagging) |
    | [随机森林](../appendix/glossary.md#g-随机森林) | 树模型与集成 | [Bagging](../appendix/glossary.md#g-Bagging) |
    | [袋外误差](../appendix/glossary.md#g-袋外误差) | 树模型与集成 | [随机森林](../appendix/glossary.md#g-随机森林)、[自助采样](../appendix/glossary.md#g-自助采样) |
    | [特征重要性](../appendix/glossary.md#g-特征重要性) | 树模型与集成 | [随机森林](../appendix/glossary.md#g-随机森林) |
    | [弱学习器](../appendix/glossary.md#g-弱学习器) | 树模型与集成 | [集成学习](../appendix/glossary.md#g-集成学习) |
    | [AdaBoost](../appendix/glossary.md#g-AdaBoost) | 树模型与集成 | [Boosting](../appendix/glossary.md#g-Boosting) |
    | [梯度提升树](../appendix/glossary.md#g-梯度提升树) | 树模型与集成 | [Boosting](../appendix/glossary.md#g-Boosting)、[梯度下降](../appendix/glossary.md#g-梯度下降)、[回归树](../appendix/glossary.md#g-回归树)、[残差](../appendix/glossary.md#g-残差) |
    | [早停](../appendix/glossary.md#g-早停) | 神经网络 | [验证集](../appendix/glossary.md#g-验证集)、[过拟合](../appendix/glossary.md#g-过拟合) |
    | [XGBoost](../appendix/glossary.md#g-XGBoost) | 树模型与集成 | [梯度提升树](../appendix/glossary.md#g-梯度提升树)、[正则化](../appendix/glossary.md#g-正则化) |
    | [LightGBM](../appendix/glossary.md#g-LightGBM) | 树模型与集成 | [梯度提升树](../appendix/glossary.md#g-梯度提升树) |

??? note "第 6 章 支持向量机（16 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [支持向量机](../appendix/glossary.md#g-支持向量机) | 分类 | [决策边界](../appendix/glossary.md#g-决策边界) |
    | [线性可分](../appendix/glossary.md#g-线性可分) | 分类 | [决策边界](../appendix/glossary.md#g-决策边界) |
    | [间隔](../appendix/glossary.md#g-间隔) | 分类 | [支持向量机](../appendix/glossary.md#g-支持向量机) |
    | [支持向量](../appendix/glossary.md#g-支持向量) | 分类 | [间隔](../appendix/glossary.md#g-间隔) |
    | [硬间隔](../appendix/glossary.md#g-硬间隔) | 分类 | [间隔](../appendix/glossary.md#g-间隔) |
    | [软间隔](../appendix/glossary.md#g-软间隔) | 分类 | [间隔](../appendix/glossary.md#g-间隔) |
    | [合页损失](../appendix/glossary.md#g-合页损失) | 分类 | [软间隔](../appendix/glossary.md#g-软间隔)、[损失函数](../appendix/glossary.md#g-损失函数) |
    | [惩罚参数 C](../appendix/glossary.md#g-惩罚参数-C) | 分类 | [硬间隔](../appendix/glossary.md#g-硬间隔)、[软间隔](../appendix/glossary.md#g-软间隔)、[正则化](../appendix/glossary.md#g-正则化) |
    | [核函数](../appendix/glossary.md#g-核函数) | 分类 | [支持向量机](../appendix/glossary.md#g-支持向量机)、[线性可分](../appendix/glossary.md#g-线性可分) |
    | [核技巧](../appendix/glossary.md#g-核技巧) | 分类 | [核函数](../appendix/glossary.md#g-核函数) |
    | [线性核](../appendix/glossary.md#g-线性核) | 分类 | [核函数](../appendix/glossary.md#g-核函数) |
    | [多项式核](../appendix/glossary.md#g-多项式核) | 分类 | [核函数](../appendix/glossary.md#g-核函数) |
    | [RBF 核](../appendix/glossary.md#g-RBF-核) | 分类 | [核函数](../appendix/glossary.md#g-核函数) |
    | [gamma 参数](../appendix/glossary.md#g-gamma-参数) | 分类 | [RBF 核](../appendix/glossary.md#g-RBF-核) |
    | [支持向量回归](../appendix/glossary.md#g-支持向量回归) | 分类 | [支持向量机](../appendix/glossary.md#g-支持向量机)、[回归](../appendix/glossary.md#g-回归) |
    | [一对一](../appendix/glossary.md#g-一对一) | 分类 | [多分类](../appendix/glossary.md#g-多分类)、[支持向量机](../appendix/glossary.md#g-支持向量机) |

??? note "第 7 章 KNN、贝叶斯与关联规则（25 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [K 近邻](../appendix/glossary.md#g-K-近邻) | 分类 | [分类](../appendix/glossary.md#g-分类)、[特征](../appendix/glossary.md#g-特征) |
    | [距离度量](../appendix/glossary.md#g-距离度量) | 分类 | [K 近邻](../appendix/glossary.md#g-K-近邻)、[标准化](../appendix/glossary.md#g-标准化) |
    | [欧氏距离](../appendix/glossary.md#g-欧氏距离) | 分类 | [距离度量](../appendix/glossary.md#g-距离度量) |
    | [曼哈顿距离](../appendix/glossary.md#g-曼哈顿距离) | 分类 | [距离度量](../appendix/glossary.md#g-距离度量) |
    | [余弦相似度](../appendix/glossary.md#g-余弦相似度) | 分类 | [距离度量](../appendix/glossary.md#g-距离度量) |
    | [K 值选择](../appendix/glossary.md#g-K-值选择) | 分类 | [K 近邻](../appendix/glossary.md#g-K-近邻)、[过拟合](../appendix/glossary.md#g-过拟合)、[欠拟合](../appendix/glossary.md#g-欠拟合) |
    | [维度灾难](../appendix/glossary.md#g-维度灾难) | 分类 | [K 近邻](../appendix/glossary.md#g-K-近邻) |
    | [条件概率](../appendix/glossary.md#g-条件概率) | 分类 | — |
    | [贝叶斯定理](../appendix/glossary.md#g-贝叶斯定理) | 分类 | [条件概率](../appendix/glossary.md#g-条件概率) |
    | [先验概率与后验概率](../appendix/glossary.md#g-先验概率与后验概率) | 分类 | [贝叶斯定理](../appendix/glossary.md#g-贝叶斯定理) |
    | [朴素贝叶斯](../appendix/glossary.md#g-朴素贝叶斯) | 分类 | [贝叶斯定理](../appendix/glossary.md#g-贝叶斯定理)、[分类](../appendix/glossary.md#g-分类) |
    | [条件独立假设](../appendix/glossary.md#g-条件独立假设) | 分类 | [朴素贝叶斯](../appendix/glossary.md#g-朴素贝叶斯) |
    | [拉普拉斯平滑](../appendix/glossary.md#g-拉普拉斯平滑) | 分类 | [朴素贝叶斯](../appendix/glossary.md#g-朴素贝叶斯) |
    | [高斯朴素贝叶斯](../appendix/glossary.md#g-高斯朴素贝叶斯) | 分类 | [朴素贝叶斯](../appendix/glossary.md#g-朴素贝叶斯)、[数值型特征](../appendix/glossary.md#g-数值型特征) |
    | [多项式朴素贝叶斯](../appendix/glossary.md#g-多项式朴素贝叶斯) | 分类 | [朴素贝叶斯](../appendix/glossary.md#g-朴素贝叶斯) |
    | [关联规则](../appendix/glossary.md#g-关联规则) | 无监督学习 | [无监督学习](../appendix/glossary.md#g-无监督学习)、[条件概率](../appendix/glossary.md#g-条件概率) |
    | [项集](../appendix/glossary.md#g-项集) | 无监督学习 | [关联规则](../appendix/glossary.md#g-关联规则) |
    | [支持度](../appendix/glossary.md#g-支持度) | 无监督学习 | [项集](../appendix/glossary.md#g-项集) |
    | [频繁项集](../appendix/glossary.md#g-频繁项集) | 无监督学习 | [项集](../appendix/glossary.md#g-项集)、[支持度](../appendix/glossary.md#g-支持度) |
    | [置信度](../appendix/glossary.md#g-置信度) | 无监督学习 | [支持度](../appendix/glossary.md#g-支持度)、[条件概率](../appendix/glossary.md#g-条件概率) |
    | [提升度](../appendix/glossary.md#g-提升度) | 无监督学习 | [置信度](../appendix/glossary.md#g-置信度) |
    | [Apriori 算法](../appendix/glossary.md#g-Apriori-算法) | 无监督学习 | [频繁项集](../appendix/glossary.md#g-频繁项集) |
    | [FP-Growth 算法](../appendix/glossary.md#g-FP-Growth-算法) | 无监督学习 | [频繁项集](../appendix/glossary.md#g-频繁项集) |
    | [购物篮分析](../appendix/glossary.md#g-购物篮分析) | 无监督学习 | [关联规则](../appendix/glossary.md#g-关联规则) |
    | [推荐系统](../appendix/glossary.md#g-推荐系统) | 无监督学习 | [购物篮分析](../appendix/glossary.md#g-购物篮分析) |

??? note "第 8 章 聚类与降维（17 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [聚类](../appendix/glossary.md#g-聚类) | 基础概念 | [无监督学习](../appendix/glossary.md#g-无监督学习)、[距离度量](../appendix/glossary.md#g-距离度量) |
    | [K 均值](../appendix/glossary.md#g-K-均值) | 无监督学习 | [聚类](../appendix/glossary.md#g-聚类)、[欧氏距离](../appendix/glossary.md#g-欧氏距离) |
    | [簇中心](../appendix/glossary.md#g-簇中心) | 无监督学习 | [K 均值](../appendix/glossary.md#g-K-均值) |
    | [肘部法则](../appendix/glossary.md#g-肘部法则) | 无监督学习 | [K 均值](../appendix/glossary.md#g-K-均值) |
    | [轮廓系数](../appendix/glossary.md#g-轮廓系数) | 无监督学习 | [聚类](../appendix/glossary.md#g-聚类) |
    | [层次聚类](../appendix/glossary.md#g-层次聚类) | 无监督学习 | [聚类](../appendix/glossary.md#g-聚类) |
    | [树状图](../appendix/glossary.md#g-树状图) | 无监督学习 | [层次聚类](../appendix/glossary.md#g-层次聚类) |
    | [DBSCAN](../appendix/glossary.md#g-DBSCAN) | 无监督学习 | [聚类](../appendix/glossary.md#g-聚类) |
    | [密度可达](../appendix/glossary.md#g-密度可达) | 无监督学习 | [DBSCAN](../appendix/glossary.md#g-DBSCAN) |
    | [噪声点](../appendix/glossary.md#g-噪声点) | 无监督学习 | [DBSCAN](../appendix/glossary.md#g-DBSCAN)、[异常值](../appendix/glossary.md#g-异常值) |
    | [客户分群](../appendix/glossary.md#g-客户分群) | 无监督学习 | [K 均值](../appendix/glossary.md#g-K-均值) |
    | [降维](../appendix/glossary.md#g-降维) | 无监督学习 | [维度灾难](../appendix/glossary.md#g-维度灾难) |
    | [主成分分析](../appendix/glossary.md#g-主成分分析) | 无监督学习 | [降维](../appendix/glossary.md#g-降维)、[标准化](../appendix/glossary.md#g-标准化) |
    | [方差解释率](../appendix/glossary.md#g-方差解释率) | 无监督学习 | [主成分分析](../appendix/glossary.md#g-主成分分析) |
    | [t-SNE](../appendix/glossary.md#g-t-SNE) | 无监督学习 | [降维](../appendix/glossary.md#g-降维) |
    | [异常检测](../appendix/glossary.md#g-异常检测) | 无监督学习 | [聚类](../appendix/glossary.md#g-聚类)、[异常值](../appendix/glossary.md#g-异常值) |
    | [孤立森林](../appendix/glossary.md#g-孤立森林) | 无监督学习 | [异常检测](../appendix/glossary.md#g-异常检测)、[随机森林](../appendix/glossary.md#g-随机森林) |

??? note "第 9 章 调优与部署（21 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [偏差与方差](../appendix/glossary.md#g-偏差与方差) | 评估与调优 | [过拟合](../appendix/glossary.md#g-过拟合)、[欠拟合](../appendix/glossary.md#g-欠拟合) |
    | [交叉验证](../appendix/glossary.md#g-交叉验证) | 评估与调优 | [验证集](../appendix/glossary.md#g-验证集) |
    | [K 折交叉验证](../appendix/glossary.md#g-K-折交叉验证) | 评估与调优 | [交叉验证](../appendix/glossary.md#g-交叉验证) |
    | [分层抽样](../appendix/glossary.md#g-分层抽样) | 评估与调优 | [K 折交叉验证](../appendix/glossary.md#g-K-折交叉验证)、[类别不平衡](../appendix/glossary.md#g-类别不平衡) |
    | [时间序列交叉验证](../appendix/glossary.md#g-时间序列交叉验证) | 评估与调优 | [K 折交叉验证](../appendix/glossary.md#g-K-折交叉验证) |
    | [学习曲线](../appendix/glossary.md#g-学习曲线) | 评估与调优 | [偏差与方差](../appendix/glossary.md#g-偏差与方差) |
    | [验证曲线](../appendix/glossary.md#g-验证曲线) | 评估与调优 | [学习曲线](../appendix/glossary.md#g-学习曲线) |
    | [网格搜索](../appendix/glossary.md#g-网格搜索) | 评估与调优 | [超参数](../appendix/glossary.md#g-超参数)、[交叉验证](../appendix/glossary.md#g-交叉验证) |
    | [随机搜索](../appendix/glossary.md#g-随机搜索) | 评估与调优 | [网格搜索](../appendix/glossary.md#g-网格搜索) |
    | [Optuna 调参](../appendix/glossary.md#g-Optuna-调参) | 评估与调优 | [随机搜索](../appendix/glossary.md#g-随机搜索) |
    | [流水线](../appendix/glossary.md#g-流水线) | 评估与调优 | [数据泄露](../appendix/glossary.md#g-数据泄露)、[标准化](../appendix/glossary.md#g-标准化) |
    | [列转换器](../appendix/glossary.md#g-列转换器) | 评估与调优 | [流水线](../appendix/glossary.md#g-流水线)、[独热编码](../appendix/glossary.md#g-独热编码) |
    | [置换重要性](../appendix/glossary.md#g-置换重要性) | 评估与调优 | [特征重要性](../appendix/glossary.md#g-特征重要性) |
    | [SHAP 值](../appendix/glossary.md#g-SHAP-值) | 评估与调优 | [特征重要性](../appendix/glossary.md#g-特征重要性) |
    | [模型保存](../appendix/glossary.md#g-模型保存) | 强化学习与责任 | [模型](../appendix/glossary.md#g-模型) |
    | [joblib](../appendix/glossary.md#g-joblib) | 强化学习与责任 | [模型保存](../appendix/glossary.md#g-模型保存) |
    | [模型接口](../appendix/glossary.md#g-模型接口) | 强化学习与责任 | [模型保存](../appendix/glossary.md#g-模型保存) |
    | [FastAPI](../appendix/glossary.md#g-FastAPI) | 强化学习与责任 | [模型接口](../appendix/glossary.md#g-模型接口) |
    | [Gradio](../appendix/glossary.md#g-Gradio) | 强化学习与责任 | [模型接口](../appendix/glossary.md#g-模型接口) |
    | [模型部署](../appendix/glossary.md#g-模型部署) | 强化学习与责任 | [模型接口](../appendix/glossary.md#g-模型接口) |
    | [数据漂移](../appendix/glossary.md#g-数据漂移) | 评估与调优 | [模型部署](../appendix/glossary.md#g-模型部署) |

??? note "第 10 章 神经网络（22 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [神经网络](../appendix/glossary.md#g-神经网络) | 神经网络 | [逻辑回归](../appendix/glossary.md#g-逻辑回归) |
    | [神经元](../appendix/glossary.md#g-神经元) | 神经网络 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [感知机](../appendix/glossary.md#g-感知机) | 神经网络 | [神经元](../appendix/glossary.md#g-神经元)、[决策边界](../appendix/glossary.md#g-决策边界) |
    | [异或问题](../appendix/glossary.md#g-异或问题) | 神经网络 | [感知机](../appendix/glossary.md#g-感知机)、[线性可分](../appendix/glossary.md#g-线性可分) |
    | [多层感知机](../appendix/glossary.md#g-多层感知机) | 神经网络 | [感知机](../appendix/glossary.md#g-感知机)、[异或问题](../appendix/glossary.md#g-异或问题) |
    | [激活函数](../appendix/glossary.md#g-激活函数) | 神经网络 | [神经元](../appendix/glossary.md#g-神经元)、[Sigmoid 函数](../appendix/glossary.md#g-Sigmoid-函数) |
    | [ReLU](../appendix/glossary.md#g-ReLU) | 神经网络 | [激活函数](../appendix/glossary.md#g-激活函数) |
    | [隐藏层](../appendix/glossary.md#g-隐藏层) | 神经网络 | [多层感知机](../appendix/glossary.md#g-多层感知机) |
    | [前向传播](../appendix/glossary.md#g-前向传播) | 神经网络 | [多层感知机](../appendix/glossary.md#g-多层感知机) |
    | [反向传播](../appendix/glossary.md#g-反向传播) | 神经网络 | [前向传播](../appendix/glossary.md#g-前向传播)、[梯度下降](../appendix/glossary.md#g-梯度下降) |
    | [链式法则](../appendix/glossary.md#g-链式法则) | 神经网络 | [反向传播](../appendix/glossary.md#g-反向传播) |
    | [梯度消失](../appendix/glossary.md#g-梯度消失) | 神经网络 | [反向传播](../appendix/glossary.md#g-反向传播)、[激活函数](../appendix/glossary.md#g-激活函数) |
    | [优化器](../appendix/glossary.md#g-优化器) | 神经网络 | [小批量梯度下降](../appendix/glossary.md#g-小批量梯度下降) |
    | [Adam](../appendix/glossary.md#g-Adam) | 神经网络 | [优化器](../appendix/glossary.md#g-优化器) |
    | [轮次与批大小](../appendix/glossary.md#g-轮次与批大小) | 神经网络 | [小批量梯度下降](../appendix/glossary.md#g-小批量梯度下降) |
    | [Dropout](../appendix/glossary.md#g-Dropout) | 神经网络 | [过拟合](../appendix/glossary.md#g-过拟合)、[隐藏层](../appendix/glossary.md#g-隐藏层) |
    | [批归一化](../appendix/glossary.md#g-批归一化) | 神经网络 | [标准化](../appendix/glossary.md#g-标准化)、[隐藏层](../appendix/glossary.md#g-隐藏层) |
    | [PyTorch](../appendix/glossary.md#g-PyTorch) | 神经网络 | — |
    | [张量](../appendix/glossary.md#g-张量) | 神经网络 | [PyTorch](../appendix/glossary.md#g-PyTorch) |
    | [自动求导](../appendix/glossary.md#g-自动求导) | 神经网络 | [张量](../appendix/glossary.md#g-张量)、[反向传播](../appendix/glossary.md#g-反向传播) |
    | [Keras](../appendix/glossary.md#g-Keras) | 神经网络 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [TensorFlow](../appendix/glossary.md#g-TensorFlow) | 神经网络 | [Keras](../appendix/glossary.md#g-Keras) |

??? note "第 11 章 CNN 与 RNN（21 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [卷积神经网络](../appendix/glossary.md#g-卷积神经网络) | 神经网络 | [多层感知机](../appendix/glossary.md#g-多层感知机) |
    | [图像的数字表示](../appendix/glossary.md#g-图像的数字表示) | 神经网络 | [张量](../appendix/glossary.md#g-张量) |
    | [卷积](../appendix/glossary.md#g-卷积) | 神经网络 | [卷积神经网络](../appendix/glossary.md#g-卷积神经网络)、[图像的数字表示](../appendix/glossary.md#g-图像的数字表示) |
    | [卷积核](../appendix/glossary.md#g-卷积核) | 神经网络 | [卷积](../appendix/glossary.md#g-卷积) |
    | [特征图](../appendix/glossary.md#g-特征图) | 神经网络 | [卷积核](../appendix/glossary.md#g-卷积核) |
    | [步长与填充](../appendix/glossary.md#g-步长与填充) | 神经网络 | [卷积](../appendix/glossary.md#g-卷积) |
    | [池化](../appendix/glossary.md#g-池化) | 神经网络 | [特征图](../appendix/glossary.md#g-特征图) |
    | [全连接层](../appendix/glossary.md#g-全连接层) | 神经网络 | [多层感知机](../appendix/glossary.md#g-多层感知机) |
    | [LeNet](../appendix/glossary.md#g-LeNet) | 神经网络 | [卷积](../appendix/glossary.md#g-卷积)、[池化](../appendix/glossary.md#g-池化)、[全连接层](../appendix/glossary.md#g-全连接层) |
    | [数据增强](../appendix/glossary.md#g-数据增强) | 神经网络 | [卷积神经网络](../appendix/glossary.md#g-卷积神经网络)、[过拟合](../appendix/glossary.md#g-过拟合) |
    | [迁移学习](../appendix/glossary.md#g-迁移学习) | 神经网络 | [卷积神经网络](../appendix/glossary.md#g-卷积神经网络) |
    | [预训练模型](../appendix/glossary.md#g-预训练模型) | 神经网络 | [迁移学习](../appendix/glossary.md#g-迁移学习) |
    | [循环神经网络](../appendix/glossary.md#g-循环神经网络) | 序列与时间序列 | [多层感知机](../appendix/glossary.md#g-多层感知机) |
    | [序列数据](../appendix/glossary.md#g-序列数据) | 序列与时间序列 | [循环神经网络](../appendix/glossary.md#g-循环神经网络) |
    | [隐藏状态](../appendix/glossary.md#g-隐藏状态) | 序列与时间序列 | [循环神经网络](../appendix/glossary.md#g-循环神经网络) |
    | [时间步](../appendix/glossary.md#g-时间步) | 序列与时间序列 | [序列数据](../appendix/glossary.md#g-序列数据) |
    | [长期依赖](../appendix/glossary.md#g-长期依赖) | 序列与时间序列 | [隐藏状态](../appendix/glossary.md#g-隐藏状态)、[梯度消失](../appendix/glossary.md#g-梯度消失) |
    | [LSTM](../appendix/glossary.md#g-LSTM) | 序列与时间序列 | [长期依赖](../appendix/glossary.md#g-长期依赖) |
    | [门控机制](../appendix/glossary.md#g-门控机制) | 序列与时间序列 | [LSTM](../appendix/glossary.md#g-LSTM) |
    | [GRU](../appendix/glossary.md#g-GRU) | 序列与时间序列 | [门控机制](../appendix/glossary.md#g-门控机制) |
    | [词嵌入](../appendix/glossary.md#g-词嵌入) | 序列与时间序列 | [序列数据](../appendix/glossary.md#g-序列数据) |

??? note "第 12 章 时序与强化学习（28 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [时间序列](../appendix/glossary.md#g-时间序列) | 序列与时间序列 | [序列数据](../appendix/glossary.md#g-序列数据) |
    | [趋势](../appendix/glossary.md#g-趋势) | 序列与时间序列 | [时间序列](../appendix/glossary.md#g-时间序列) |
    | [季节性](../appendix/glossary.md#g-季节性) | 序列与时间序列 | [时间序列](../appendix/glossary.md#g-时间序列) |
    | [平稳性](../appendix/glossary.md#g-平稳性) | 序列与时间序列 | [趋势](../appendix/glossary.md#g-趋势)、[季节性](../appendix/glossary.md#g-季节性) |
    | [差分](../appendix/glossary.md#g-差分) | 序列与时间序列 | [平稳性](../appendix/glossary.md#g-平稳性) |
    | [滞后特征](../appendix/glossary.md#g-滞后特征) | 序列与时间序列 | [时间序列](../appendix/glossary.md#g-时间序列)、[特征构造](../appendix/glossary.md#g-特征构造) |
    | [滑动窗口](../appendix/glossary.md#g-滑动窗口) | 序列与时间序列 | [滞后特征](../appendix/glossary.md#g-滞后特征) |
    | [移动平均](../appendix/glossary.md#g-移动平均) | 序列与时间序列 | [时间序列](../appendix/glossary.md#g-时间序列) |
    | [指数平滑](../appendix/glossary.md#g-指数平滑) | 序列与时间序列 | [移动平均](../appendix/glossary.md#g-移动平均) |
    | [ARIMA](../appendix/glossary.md#g-ARIMA) | 序列与时间序列 | [差分](../appendix/glossary.md#g-差分)、[移动平均](../appendix/glossary.md#g-移动平均) |
    | [自相关](../appendix/glossary.md#g-自相关) | 序列与时间序列 | [时间序列](../appendix/glossary.md#g-时间序列)、[相关系数](../appendix/glossary.md#g-相关系数) |
    | [基于特征的时序预测](../appendix/glossary.md#g-基于特征的时序预测) | 序列与时间序列 | [滑动窗口](../appendix/glossary.md#g-滑动窗口)、[LightGBM](../appendix/glossary.md#g-LightGBM) |
    | [LSTM 时序预测](../appendix/glossary.md#g-LSTM-时序预测) | 序列与时间序列 | [LSTM](../appendix/glossary.md#g-LSTM)、[滑动窗口](../appendix/glossary.md#g-滑动窗口) |
    | [时序评价指标](../appendix/glossary.md#g-时序评价指标) | 序列与时间序列 | [平均绝对误差](../appendix/glossary.md#g-平均绝对误差)、[时间序列交叉验证](../appendix/glossary.md#g-时间序列交叉验证) |
    | [智能体](../appendix/glossary.md#g-智能体) | 强化学习与责任 | [强化学习](../appendix/glossary.md#g-强化学习) |
    | [环境](../appendix/glossary.md#g-环境) | 强化学习与责任 | [智能体](../appendix/glossary.md#g-智能体) |
    | [状态](../appendix/glossary.md#g-状态) | 强化学习与责任 | [环境](../appendix/glossary.md#g-环境) |
    | [动作](../appendix/glossary.md#g-动作) | 强化学习与责任 | [智能体](../appendix/glossary.md#g-智能体) |
    | [奖励](../appendix/glossary.md#g-奖励) | 强化学习与责任 | [动作](../appendix/glossary.md#g-动作)、[状态](../appendix/glossary.md#g-状态) |
    | [策略](../appendix/glossary.md#g-策略) | 强化学习与责任 | [状态](../appendix/glossary.md#g-状态)、[动作](../appendix/glossary.md#g-动作) |
    | [探索与利用](../appendix/glossary.md#g-探索与利用) | 强化学习与责任 | [策略](../appendix/glossary.md#g-策略) |
    | [马尔可夫决策过程](../appendix/glossary.md#g-马尔可夫决策过程) | 强化学习与责任 | [状态](../appendix/glossary.md#g-状态)、[动作](../appendix/glossary.md#g-动作)、[奖励](../appendix/glossary.md#g-奖励) |
    | [Q 学习](../appendix/glossary.md#g-Q-学习) | 强化学习与责任 | [马尔可夫决策过程](../appendix/glossary.md#g-马尔可夫决策过程)、[探索与利用](../appendix/glossary.md#g-探索与利用) |
    | [Q 表](../appendix/glossary.md#g-Q-表) | 强化学习与责任 | [Q 学习](../appendix/glossary.md#g-Q-学习) |
    | [深度强化学习](../appendix/glossary.md#g-深度强化学习) | 强化学习与责任 | [Q 学习](../appendix/glossary.md#g-Q-学习)、[神经网络](../appendix/glossary.md#g-神经网络) |
    | [人类反馈强化学习](../appendix/glossary.md#g-人类反馈强化学习) | 强化学习与责任 | [深度强化学习](../appendix/glossary.md#g-深度强化学习)、[深度学习](../appendix/glossary.md#g-深度学习) |
    | [算法公平性](../appendix/glossary.md#g-算法公平性) | 强化学习与责任 | [特征重要性](../appendix/glossary.md#g-特征重要性) |
    | [数据合规](../appendix/glossary.md#g-数据合规) | 强化学习与责任 | [数据清洗规范](../appendix/glossary.md#g-数据清洗规范) |

<!-- graph:end -->
