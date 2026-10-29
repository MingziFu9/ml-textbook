"""生成本书使用的模拟数据集（固定随机种子，重复运行结果相同）。

真实的银行贷款、电信客户数据涉及个人信息，不能公开。这里按公开报道和行业常识中的
大致分布规律生成模拟数据，字段含义和真实业务一致，只用于教学。

用法：python make_datasets.py
输出：
    loan_apply.csv     贷款审批，2000 行（第 1、5 章）
    telco_churn.csv    电信客户流失，5000 行（第 4、5、9 章）
    house_raw.csv      广州二手房挂牌数据（未清洗，故意带缺失、重复、异常和格式问题），约 3000 行（第 2、3、9 章）
    card_fraud.csv     信用卡交易，欺诈约占 1%，20000 行（第 4、8 章）
"""
import numpy as np
import pandas as pd

SEED = 2026


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_loan(n=2000, seed=SEED):
    """贷款审批：是否批准由收入、负债、房产、信用记录等共同决定，带少量人工审批的随机性。"""
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 61, n)
    work_years = np.clip(age - 20 - rng.integers(0, 8, n), 0, None)
    income = np.round(rng.lognormal(np.log(6000), 0.45, n) * (1 + work_years * 0.015), -2)
    has_house = rng.random(n) < sigmoid((age - 35) / 6)
    debt_ratio = np.round(np.clip(rng.beta(2, 5, n) + (~has_house) * 0.05, 0, 0.95), 2)
    credit = rng.choice(["一般", "好", "非常好"], n, p=[0.35, 0.45, 0.20])
    overdue = rng.poisson(np.where(credit == "一般", 1.5, np.where(credit == "好", 0.4, 0.1)))
    amount = np.round(rng.uniform(1, 30, n), 1)  # 申请金额，万元

    score = (
        1.2 * (income > 8000) + 0.8 * (income > 5000)
        + 1.5 * has_house
        + np.select([credit == "一般", credit == "好"], [-1.0, 0.3], 1.0)
        - 2.0 * (debt_ratio > 0.5) - 0.6 * overdue
        - 0.04 * amount + 0.03 * np.minimum(work_years, 10)
        - 0.3
    )
    approved = rng.random(n) < sigmoid(2.0 * score)
    return pd.DataFrame({
        "年龄": age,
        "工作年限": work_years,
        "月收入": income.astype(int),
        "有房产": np.where(has_house, "是", "否"),
        "负债率": debt_ratio,
        "信用等级": credit,
        "近两年逾期次数": overdue,
        "申请金额_万元": amount,
        "是否批准": np.where(approved, "是", "否"),
    })


def make_churn(n=5000, seed=SEED + 1):
    """电信客户流失：月租高、合约短、在网时间短、无增值服务、投诉多的客户更容易流失。"""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["月付", "一年", "两年"], n, p=[0.55, 0.25, 0.20])
    tenure = np.where(
        contract == "月付", rng.integers(1, 40, n),
        np.where(contract == "一年", rng.integers(6, 60, n), rng.integers(12, 73, n)),
    )
    internet = rng.choice(["光纤", "DSL", "无"], n, p=[0.45, 0.35, 0.20])
    base = np.select([internet == "光纤", internet == "DSL"], [70, 50], 20)
    tech_support = np.where(internet == "无", "无网络", rng.choice(["是", "否"], n, p=[0.35, 0.65]))
    streaming = np.where(internet == "无", "无网络", rng.choice(["是", "否"], n, p=[0.45, 0.55]))
    monthly = np.round(base + (streaming == "是") * 15 + (tech_support == "是") * 8 + rng.normal(0, 6, n), 1)
    payment = rng.choice(["自动扣款", "手机支付", "营业厅缴费"], n, p=[0.45, 0.40, 0.15])
    complaints = rng.poisson(np.where(internet == "光纤", 0.8, 0.4))
    age = rng.integers(18, 76, n)
    senior = age >= 60
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 1)

    z = (
        -1.3
        + np.select([contract == "月付", contract == "一年"], [1.4, 0.0], -1.2)
        - 0.045 * tenure
        + 0.025 * (monthly - 60)
        + 0.6 * (internet == "光纤")
        - 0.7 * (tech_support == "是")
        + 0.45 * (payment == "营业厅缴费")
        + 0.35 * senior
        + 0.55 * complaints
        # 非线性：月付且新入网、投诉多的客户流失明显更高（树模型更容易学到）
        + 1.0 * ((contract == "月付") & (tenure <= 6))
        + 0.8 * (complaints >= 3)
    )
    churn = rng.random(n) < sigmoid(z)
    return pd.DataFrame({
        "客户编号": [f"C{100000 + i}" for i in range(n)],
        "年龄": age,
        "在网月数": tenure,
        "合约类型": contract,
        "宽带类型": internet,
        "技术支持": tech_support,
        "视频会员": streaming,
        "付款方式": payment,
        "月消费": monthly,
        "累计消费": total,
        "近半年投诉次数": complaints,
        "是否流失": np.where(churn, "是", "否"),
    })


DISTRICTS = {  # 区域: (挂牌占比, 基准单价 元/平方米, 平均建成年份)，大致参照 2025 年广州各区二手房挂牌情况
    "天河": (0.14, 62000, 2006), "越秀": (0.09, 55000, 1998), "海珠": (0.12, 45000, 2004), "荔湾": (0.07, 38000, 2001),
    "白云": (0.13, 33000, 2007), "黄埔": (0.09, 33000, 2011), "番禺": (0.16, 30000, 2010), "南沙": (0.06, 18000, 2015),
    "花都": (0.07, 16000, 2012), "增城": (0.07, 15000, 2013),
}


def make_house(n=3000, seed=SEED + 2):
    """二手房挂牌数据。总价 = 单价 × 面积，单价由区域、房龄、地铁距离、装修、楼层、朝向决定，再乘随机波动。
    为了练习数据清洗，故意加入：缺失值、重复行、面积写法不统一、装修写法不统一、录入错误造成的异常值。"""
    rng = np.random.default_rng(seed)
    names = list(DISTRICTS)
    district = rng.choice(names, n, p=[DISTRICTS[d][0] for d in names])
    base = np.array([DISTRICTS[d][1] for d in district])
    # 小区：每个区约 40 个小区，每个小区有自己的单价水平（地段、物业、学位等），挂牌数量多少不一
    heads = list("锦绣碧桂翠华金汇雅逸丽景御龙凤云天星海珠岭湖")
    tails = ["花园", "苑", "公馆", "豪庭", "新村", "山庄", "广场", "雅居"]
    estates = {d: [f"{heads[(i * 7 + k) % len(heads)]}{heads[(i * 3 + k * 5 + 1) % len(heads)]}{tails[(i + k) % len(tails)]}"
                   for i in range(40)] for k, d in enumerate(names)}
    estate_effect = {(d, e): rng.lognormal(0, 0.12) for d in names for e in estates[d]}
    popularity = rng.dirichlet(np.ones(40) * 0.8, len(names))
    estate = np.array([estates[d][rng.choice(40, p=popularity[names.index(d)])] for d in district])
    area = np.round(np.clip(rng.lognormal(np.log(88), 0.32, n), 32, 260), 1)
    rooms = np.clip(np.round(area / 32 + rng.normal(0, 0.5, n)), 1, 6).astype(int)
    halls = np.where(rooms >= 3, 2, 1)
    year = np.clip(np.round(rng.normal([DISTRICTS[d][2] for d in district], 6)), 1985, 2023).astype(int)
    total_floors = rng.choice([7, 9, 11, 18, 24, 28, 32, 33], n, p=[0.08, 0.08, 0.1, 0.2, 0.16, 0.14, 0.12, 0.12])
    floor_pos = rng.choice(["低楼层", "中楼层", "高楼层"], n, p=[0.3, 0.4, 0.3])
    decoration = rng.choice(["精装", "简装", "毛坯"], n, p=[0.55, 0.35, 0.10])
    facing = rng.choice(["南", "南北", "东南", "东", "西南", "西", "北"], n, p=[0.25, 0.25, 0.15, 0.1, 0.1, 0.08, 0.07])
    subway = np.round(np.clip(rng.exponential(900, n) + 100, 100, 6000), -1)
    listed = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 731, n), unit="D")

    unit = (base
            * np.array([estate_effect[(d, e)] for d, e in zip(district, estate)])
            * (1 - 0.007 * (2023 - year))
            * (1 - 0.06 * np.minimum(subway, 3000) / 1000)
            * np.select([decoration == "精装", decoration == "毛坯"], [1.06, 0.95], 1.0)
            * np.select([floor_pos == "中楼层", floor_pos == "低楼层"], [1.02, 0.97], 1.01)
            * np.where(np.isin(facing, ["南", "南北"]), 1.03, 1.0)
            * (1 - 0.0045 * (listed - pd.Timestamp("2024-01-01")).days / 30)   # 两年里每月约降 0.45%
            * rng.lognormal(0, 0.08, n))
    total = np.round(unit * area / 10000, 1)

    df = pd.DataFrame({
        "房源编号": [f"GZ{300000 + i}" for i in range(n)],
        "区域": district,
        "小区": estate,
        "面积": area.astype(object),
        "户型": [f"{r}室{h}厅" for r, h in zip(rooms, halls)],
        "楼层": [f"{p}/共{t}层" for p, t in zip(floor_pos, total_floors)],
        "建成年份": year.astype(float),
        "装修": decoration.astype(object),
        "朝向": facing,
        "距地铁站_米": subway,
        "挂牌日期": listed.strftime("%Y-%m-%d"),
        "单价_元每平方米": np.round(total * 10000 / area).astype(int),
        "总价_万元": total,
    })
    # 1. 缺失值
    df.loc[rng.random(n) < 0.07, "建成年份"] = np.nan
    df.loc[rng.random(n) < 0.05, "距地铁站_米"] = np.nan
    df.loc[rng.random(n) < 0.03, "装修"] = np.nan
    # 2. 写法不统一：面积带单位，装修多一个“修”字
    idx = rng.choice(n, 300, replace=False)
    df.loc[idx[:200], "面积"] = [f"{a}㎡" for a in area[idx[:200]]]
    df.loc[idx[200:], "面积"] = [f"{a}平米" for a in area[idx[200:]]]
    swap = rng.choice(n, 120, replace=False)
    df.loc[swap, "装修"] = df.loc[swap, "装修"].map({"精装": "精装修", "简装": "简装修", "毛坯": "毛坯"})
    # 3. 录入错误：面积多打一个 0、总价按“元”录入
    bad_area = rng.choice(n, 6, replace=False)
    df.loc[bad_area, "面积"] = [str(round(a * 10, 1)) for a in area[bad_area]]
    bad_price = rng.choice(np.setdiff1d(np.arange(n), bad_area), 5, replace=False)
    df.loc[bad_price, "总价_万元"] = df.loc[bad_price, "总价_万元"] * 10000
    # 4. 重复行：同一套房被重复抓取
    dup = df.iloc[rng.choice(n, 60, replace=False)]
    df = pd.concat([df, dup]).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["面积"] = df["面积"].astype(str)
    return df


def make_fraud(n=20000, seed=SEED + 3):
    """信用卡交易：欺诈交易金额偏大、多在深夜、境外和新设备多、离常用地远、短时间内连续交易。各特征都有重叠。"""
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < 0.01
    k = fraud.sum()

    def mix(normal, bad):
        out = normal(n)
        out[fraud] = bad(k)
        return out

    amount = np.round(mix(lambda m: rng.lognormal(np.log(120), 1.0, m), lambda m: rng.lognormal(np.log(900), 1.1, m)), 2)
    hour = mix(lambda m: np.clip(np.round(rng.normal(14, 4.5, m)), 0, 23),
               lambda m: np.where(rng.random(m) < 0.45, rng.integers(0, 6, m), rng.integers(0, 24, m))).astype(int)
    abroad = mix(lambda m: (rng.random(m) < 0.03).astype(float), lambda m: (rng.random(m) < 0.25).astype(float)).astype(int)
    new_device = mix(lambda m: (rng.random(m) < 0.06).astype(float), lambda m: (rng.random(m) < 0.40).astype(float)).astype(int)
    gap = np.round(mix(lambda m: rng.exponential(30, m), lambda m: rng.exponential(4, m)), 1)
    cnt24 = mix(lambda m: rng.poisson(2, m).astype(float), lambda m: rng.poisson(5, m).astype(float)).astype(int)
    dist = np.round(mix(lambda m: rng.exponential(6, m), lambda m: rng.exponential(120, m)), 1)
    cats = np.array(["超市", "餐饮", "网购", "加油", "数码电器", "珠宝首饰", "游戏充值"])
    merchant = np.where(fraud, rng.choice(cats, n, p=[0.08, 0.07, 0.30, 0.05, 0.22, 0.13, 0.15]),
                        rng.choice(cats, n, p=[0.28, 0.27, 0.25, 0.10, 0.05, 0.02, 0.03]))
    return pd.DataFrame({
        "交易编号": [f"T{1000000 + i}" for i in range(n)],
        "交易金额": amount,
        "交易小时": hour,
        "商户类别": merchant,
        "是否境外": abroad,
        "是否新设备": new_device,
        "距上次交易_小时": gap,
        "近24小时交易次数": cnt24,
        "距常用地_公里": dist,
        "是否欺诈": fraud.astype(int),
    })


if __name__ == "__main__":
    loan = make_loan()
    loan.to_csv("loan_apply.csv", index=False, encoding="utf-8-sig")
    churn = make_churn()
    churn.to_csv("telco_churn.csv", index=False, encoding="utf-8-sig")
    print(f"loan_apply.csv：{len(loan)} 行，批准率 {(loan['是否批准'] == '是').mean():.1%}")
    print(f"telco_churn.csv：{len(churn)} 行，流失率 {(churn['是否流失'] == '是').mean():.1%}")
    house = make_house()
    house.to_csv("house_raw.csv", index=False, encoding="utf-8-sig")
    print(f"house_raw.csv：{len(house)} 行（含重复和脏数据）")
    fraud = make_fraud()
    fraud.to_csv("card_fraud.csv", index=False, encoding="utf-8-sig")
    print(f"card_fraud.csv：{len(fraud)} 行，欺诈率 {fraud['是否欺诈'].mean():.2%}")
