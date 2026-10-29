---
title: 附录 D 术语表
description: 本书主要术语的中英文名称、简明定义和例子，按章节顺序排列
---

# 附录 D 术语表

本表按章节顺序列出本书的主要术语，每个术语给出英文名称、简明定义和例子。正文中带虚线下划线的术语，把鼠标停在上面就能看到定义，点击可以跳到本表。

通用术语（如“过拟合”“精确率”）的译名和定义以本系列教材共用的《系列公共术语表》为准，各书保持一致。

其余各章的术语随正文编写陆续补入。

## 第 5 章 决策树与集成学习

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-决策树"></span>决策树 | Decision Tree | 用一连串只看一个特征的是 / 否问题对样本逐步划分，在叶节点给出结论的模型 | “有房吗？→ 有工作吗？”两层问题决定是否批准贷款 |
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
