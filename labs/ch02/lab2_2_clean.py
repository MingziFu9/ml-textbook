"""实验 2-2 按规范清洗并编码

按《二手房数据清洗规范》逐条清洗 house_raw.csv，保存 house_clean.csv（第 3、9 章使用）；
再划分训练集和测试集，只用训练集计算填充值和标准化参数，完成编码和缩放。
运行：python lab2_2_clean.py（在 labs/ch02 目录下）
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

raw = pd.read_csv("../data/house_raw.csv")
df = raw.copy()
log = []

# 规则 1：同一房源编号只保留第一条
before = len(df)
df = df.drop_duplicates(subset="房源编号", keep="first")
log.append(("R1 去重", before - len(df)))

# 规则 2：面积去掉单位，转成数字
df["面积"] = pd.to_numeric(df["面积"].str.replace("㎡", "").str.replace("平米", ""))
log.append(("R2 面积去单位", raw["面积"].str.contains("㎡|平米").sum()))

# 规则 3：总价按“元”录入的，除以 10000；用“单价 × 面积”校验
implied = df["单价_元每平方米"] * df["面积"] / 10000           # 用单价和面积推算的总价（万元）
wrong_unit = (df["总价_万元"] > 10000) & ((df["总价_万元"] / 10000 / implied - 1).abs() < 0.05)
df.loc[wrong_unit, "总价_万元"] = df.loc[wrong_unit, "总价_万元"] / 10000
log.append(("R3 总价单位错误", wrong_unit.sum()))

# 规则 4：面积多打一个 0 的（> 300 平方米，且除以 10 后与单价推算值吻合），除以 10
implied_area = df["总价_万元"] * 10000 / df["单价_元每平方米"]
extra_zero = (df["面积"] > 300) & ((df["面积"] / 10 / implied_area - 1).abs() < 0.05)
df.loc[extra_zero, "面积"] = df.loc[extra_zero, "面积"] / 10
log.append(("R4 面积多打 0", extra_zero.sum()))

# 规则 5：装修写法统一；缺失记为“未知”
df["装修"] = df["装修"].replace({"精装修": "精装", "简装修": "简装"}).fillna("未知")
log.append(("R5 装修统一写法", raw["装修"].isin(["精装修", "简装修"]).sum()))

# 规则 6：拆分楼层和户型
df[["楼层位置", "总楼层"]] = df["楼层"].str.extract(r"(.)楼层/共(\d+)层")
df["总楼层"] = df["总楼层"].astype(int)
df[["室", "厅"]] = df["户型"].str.extract(r"(\d)室(\d)厅").astype(int)

# 规则 7：删除不能作为特征的字段：编号没有含义；单价由总价算出，预测总价时属于数据泄露
df = df.drop(columns=["楼层", "户型", "单价_元每平方米"])

print("清洗记录：")
for rule, n in log:
    print(f"  {rule:<14}{n:>5} 条")
print(f"清洗后 {len(df)} 行；仍有缺失（留到划分后再填）：{df.isna().sum()[lambda s: s > 0].to_dict()}")
print(f"总价范围 {df['总价_万元'].min()}～{df['总价_万元'].max()} 万元，面积范围 {df['面积'].min()}～{df['面积'].max()} 平方米")
df.to_csv("../data/house_clean.csv", index=False, encoding="utf-8-sig")
print("已保存 ../data/house_clean.csv")

# ---------- 以下是建模前的预处理：先划分，再用训练集的统计量处理两边 ----------
X = df.drop(columns=["房源编号", "小区", "挂牌日期", "总价_万元"])   # 小区有 300 多个取值，实验 2-3 再处理
y = df["总价_万元"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train.copy(), X_test.copy()

# 规则 8：建成年份按训练集中同区域的中位数填充；距地铁站按训练集中位数填充
year_median = X_train.groupby("区域")["建成年份"].median()
subway_median = X_train["距地铁站_米"].median()
for part in (X_train, X_test):
    part["建成年份"] = part["建成年份"].fillna(part["区域"].map(year_median))
    part["距地铁站_米"] = part["距地铁站_米"].fillna(subway_median)
print(f"\n训练集 {len(X_train)} 条，测试集 {len(X_test)} 条")
print("训练集各区建成年份中位数：", year_median.astype(int).to_dict())
print(f"训练集距地铁站中位数：{subway_median:.0f} 米")

# 规则 9：编码。楼层位置有顺序，用 0/1/2；区域、装修、朝向没有顺序，用独热编码
for part in (X_train, X_test):
    part["楼层位置"] = part["楼层位置"].map({"低": 0, "中": 1, "高": 2})
X_train = pd.get_dummies(X_train, columns=["区域", "装修", "朝向"], dtype=int)
X_test = pd.get_dummies(X_test, columns=["区域", "装修", "朝向"], dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)   # 保证两边列完全一样

# 规则 10：数值特征标准化，均值和标准差只从训练集计算
num_cols = ["面积", "建成年份", "距地铁站_米", "总楼层", "室", "厅"]
scaler = StandardScaler().fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"\n编码后共 {X_train.shape[1]} 个特征：{list(X_train.columns)}")
print("训练集标准化后的均值和标准差：")
print(X_train[num_cols].agg(["mean", "std"]).round(3).to_string())
print("测试集标准化后的均值和标准差（不必正好是 0 和 1）：")
print(X_test[num_cols].agg(["mean", "std"]).round(3).to_string())
print("\n处理后的前 2 行（只显示部分列）：")
print(X_train.iloc[:2, :10].round(2).to_string())
print(f"缺失值总数：训练集 {int(X_train.isna().sum().sum())}，测试集 {int(X_test.isna().sum().sum())}")
