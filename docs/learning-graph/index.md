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

图谱共有 **29 个概念、31 条先修关系**，概念按章排列，颜色表示 2 个类别（树模型、集成学习）。概念的定义都来自[附录 D 术语表](../appendix/glossary.md)，先修关系由编者根据各章内容整理。

## 文字版

下面按章列出每个概念的直接先修概念，便于打印或用读屏软件阅读。

??? note "第 5 章 决策树与集成学习（29 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [决策树](../appendix/glossary.md#g-决策树) | 树模型 | — |
    | [根节点](../appendix/glossary.md#g-根节点) | 树模型 | [决策树](../appendix/glossary.md#g-决策树) |
    | [叶节点](../appendix/glossary.md#g-叶节点) | 树模型 | [决策树](../appendix/glossary.md#g-决策树) |
    | [树的深度](../appendix/glossary.md#g-树的深度) | 树模型 | [决策树](../appendix/glossary.md#g-决策树) |
    | [信息熵](../appendix/glossary.md#g-信息熵) | 树模型 | [决策树](../appendix/glossary.md#g-决策树) |
    | [信息增益](../appendix/glossary.md#g-信息增益) | 树模型 | [信息熵](../appendix/glossary.md#g-信息熵) |
    | [ID3 算法](../appendix/glossary.md#g-ID3-算法) | 树模型 | [信息增益](../appendix/glossary.md#g-信息增益) |
    | [信息增益率](../appendix/glossary.md#g-信息增益率) | 树模型 | [信息增益](../appendix/glossary.md#g-信息增益) |
    | [C4.5 算法](../appendix/glossary.md#g-C4.5-算法) | 树模型 | [信息增益率](../appendix/glossary.md#g-信息增益率) |
    | [基尼系数](../appendix/glossary.md#g-基尼系数) | 树模型 | [决策树](../appendix/glossary.md#g-决策树) |
    | [CART 算法](../appendix/glossary.md#g-CART-算法) | 树模型 | [基尼系数](../appendix/glossary.md#g-基尼系数) |
    | [回归树](../appendix/glossary.md#g-回归树) | 树模型 | [CART 算法](../appendix/glossary.md#g-CART-算法) |
    | [剪枝](../appendix/glossary.md#g-剪枝) | 树模型 | [树的深度](../appendix/glossary.md#g-树的深度) |
    | [预剪枝](../appendix/glossary.md#g-预剪枝) | 树模型 | [剪枝](../appendix/glossary.md#g-剪枝) |
    | [后剪枝](../appendix/glossary.md#g-后剪枝) | 树模型 | [剪枝](../appendix/glossary.md#g-剪枝) |
    | [集成学习](../appendix/glossary.md#g-集成学习) | 集成学习 | [决策树](../appendix/glossary.md#g-决策树) |
    | [Bagging](../appendix/glossary.md#g-Bagging) | 集成学习 | [集成学习](../appendix/glossary.md#g-集成学习)、[自助采样](../appendix/glossary.md#g-自助采样) |
    | [Boosting](../appendix/glossary.md#g-Boosting) | 集成学习 | [集成学习](../appendix/glossary.md#g-集成学习)、[弱学习器](../appendix/glossary.md#g-弱学习器) |
    | [自助采样](../appendix/glossary.md#g-自助采样) | 集成学习 | — |
    | [随机森林](../appendix/glossary.md#g-随机森林) | 集成学习 | [Bagging](../appendix/glossary.md#g-Bagging) |
    | [袋外误差](../appendix/glossary.md#g-袋外误差) | 集成学习 | [随机森林](../appendix/glossary.md#g-随机森林)、[自助采样](../appendix/glossary.md#g-自助采样) |
    | [特征重要性](../appendix/glossary.md#g-特征重要性) | 集成学习 | [随机森林](../appendix/glossary.md#g-随机森林)、[基尼系数](../appendix/glossary.md#g-基尼系数) |
    | [弱学习器](../appendix/glossary.md#g-弱学习器) | 集成学习 | — |
    | [AdaBoost](../appendix/glossary.md#g-AdaBoost) | 集成学习 | [Boosting](../appendix/glossary.md#g-Boosting) |
    | [梯度提升树](../appendix/glossary.md#g-梯度提升树) | 集成学习 | [Boosting](../appendix/glossary.md#g-Boosting)、[回归树](../appendix/glossary.md#g-回归树)、[残差](../appendix/glossary.md#g-残差) |
    | [残差](../appendix/glossary.md#g-残差) | 集成学习 | — |
    | [早停](../appendix/glossary.md#g-早停) | 集成学习 | [梯度提升树](../appendix/glossary.md#g-梯度提升树) |
    | [XGBoost](../appendix/glossary.md#g-XGBoost) | 集成学习 | [梯度提升树](../appendix/glossary.md#g-梯度提升树) |
    | [LightGBM](../appendix/glossary.md#g-LightGBM) | 集成学习 | [梯度提升树](../appendix/glossary.md#g-梯度提升树) |

<!-- graph:end -->
