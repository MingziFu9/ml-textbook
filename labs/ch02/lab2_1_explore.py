"""实验 2-1 认识一份真实数据：二手房数据探索

读取未清洗的二手房挂牌数据，检查缺失、重复、格式和异常，做描述统计和可视化，为写数据质量报告收集素材。
运行：python lab2_1_explore.py（在 labs/ch02 目录下）
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# 第 1 步：读入数据，看规模、字段类型和前几行
df = pd.read_csv("../data/house_raw.csv")
print(f"共 {df.shape[0]} 行、{df.shape[1]} 列")
print(df.dtypes.to_string())
print(df.head(3).to_string())

# 第 2 步：缺失值
missing = df.isna().sum()
print("\n缺失值（只列有缺失的字段）：")
print(pd.DataFrame({"缺失数": missing, "缺失比例": (missing / len(df)).round(3)})[missing > 0].to_string())

# 第 3 步：重复值
print(f"\n完全相同的重复行：{df.duplicated().sum()} 行；房源编号重复：{df['房源编号'].duplicated().sum()} 个")

# 第 4 步：格式问题——“面积”本该是数字，却被读成了文字
print(f"\n“面积”列的类型：{df['面积'].dtype}")
not_number = df.loc[pd.to_numeric(df["面积"], errors="coerce").isna(), "面积"]
print(f"无法直接转成数字的面积：{len(not_number)} 个，例如 {not_number.head(4).tolist()}")
print("“装修”列的取值：", df["装修"].value_counts(dropna=False).to_dict())

# 第 5 步：描述统计，找异常值
area = pd.to_numeric(df["面积"].str.replace("㎡", "").str.replace("平米", ""), errors="coerce")
num = df[["建成年份", "距地铁站_米", "单价_元每平方米", "总价_万元"]].assign(面积=area)
print("\n数值字段的描述统计：")
print(num.describe().round(1).T.to_string())

# 用箱线图规则（IQR）找总价异常值
q1, q3 = df["总价_万元"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = df[(df["总价_万元"] < low) | (df["总价_万元"] > high)]
print(f"\n总价的 IQR 范围：[{low:.1f}, {high:.1f}] 万元，范围外 {len(out)} 套，其中超过 1 万万元的 {(out['总价_万元'] > 10000).sum()} 套：")
print(out.nlargest(6, "总价_万元")[["房源编号", "区域", "面积", "单价_元每平方米", "总价_万元"]].to_string(index=False))

# 第 6 步：分组统计——各区的套数和单价中位数
print("\n各区挂牌套数与单价中位数（元/平方米）：")
print(df.groupby("区域")["单价_元每平方米"].agg(["count", "median"]).sort_values("median", ascending=False).astype(int).T.to_string())

# 数据透视：区域 × 装修 的单价中位数（装修写法先临时统一）
pivot = pd.pivot_table(df.assign(装修=df["装修"].replace({"精装修": "精装", "简装修": "简装"})),
                       index="区域", columns="装修", values="单价_元每平方米", aggfunc="median")
print("\n数据透视表：各区不同装修的单价中位数（元/平方米，前 4 个区）：")
print(pivot.loc[["天河", "海珠", "番禺", "增城"], ["毛坯", "简装", "精装"]].round(-2).astype(int).to_string())

# 第 7 步：画图（先去掉明显的录入错误再画，否则图会被极端值压扁）
ok = df[(df["总价_万元"] < 10000) & (area < 1000)].assign(面积=area)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(ok["总价_万元"], bins=40, color="#3f51b5")
axes[0].set(title="总价分布", xlabel="总价（万元）", ylabel="套数")
order = ok.groupby("区域")["单价_元每平方米"].median().sort_values().index
axes[1].boxplot([ok.loc[ok["区域"] == d, "单价_元每平方米"] / 10000 for d in order], tick_labels=order)
axes[1].set(title="各区单价箱线图", ylabel="单价（万元/平方米）")
axes[1].tick_params(axis="x", rotation=45)
axes[2].scatter(ok["面积"], ok["总价_万元"], s=4, alpha=0.4)
axes[2].set(title="面积与总价", xlabel="面积（平方米）", ylabel="总价（万元）")
plt.tight_layout()
plt.savefig("lab2_1_eda.png", dpi=120)

corr = ok[["面积", "建成年份", "距地铁站_米", "单价_元每平方米", "总价_万元"]].corr().round(2)
print("\n相关系数矩阵：")
print(corr.to_string())
within = df.groupby("区域")[["建成年份", "单价_元每平方米"]].corr().xs("建成年份", level=1)["单价_元每平方米"]
print(f"各区内部“建成年份—单价”相关系数：{within.round(2).min()}～{within.round(2).max()}（全部数据合在一起是 {corr.loc['建成年份', '单价_元每平方米']}）")
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, corr.iloc[i, j], ha="center", va="center", fontsize=9)
fig.colorbar(im)
ax.set_title("相关系数热力图")
plt.tight_layout()
plt.savefig("lab2_1_corr.png", dpi=120)
print("\n图已保存：lab2_1_eda.png、lab2_1_corr.png")
