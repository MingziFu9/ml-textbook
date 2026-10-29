"""实验 1-1 环境自检脚本

检查 Python 版本、本书用到的库是否装好，并跑一个最小的机器学习例子。
运行：python check_env.py（在 labs/ch01 目录下）
"""
import importlib
import platform
import sys
import time

print(f"Python {platform.python_version()}（{platform.system()} {platform.machine()}）")
if sys.version_info < (3, 10):
    print("  ✗ Python 版本太低，请安装 3.10 或以上版本（本书用 3.11）")

# (导入名, 显示名, 是否必需)
PACKAGES = [
    ("numpy", "NumPy", True), ("pandas", "pandas", True), ("matplotlib", "Matplotlib", True),
    ("sklearn", "scikit-learn", True), ("xgboost", "XGBoost", False), ("lightgbm", "LightGBM", False),
]
missing = []
for module, name, required in PACKAGES:
    try:
        mod = importlib.import_module(module)
        print(f"  ✓ {name:<13}{mod.__version__}")
    except ImportError:
        print(f"  {'✗' if required else '-'} {name:<13}未安装{'（必需）' if required else '（第 5 章才用到，可以以后再装）'}")
        if required:
            missing.append(name)

if missing:
    print(f"\n请先安装：pip install {' '.join(m.lower() for m in missing)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)

# 中文字体：画图时中文不显示成方框
import matplotlib.font_manager as fm

fonts = {f.name for f in fm.fontManager.ttflist}
cjk = [f for f in ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Noto Sans CJK SC"] if f in fonts]
print(f"\n中文字体：{'、'.join(cjk) if cjk else '没找到，画图时中文会显示成方框，见实验 1-1 常见问题'}")

# 最小例子：训练一个模型并计时
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
start = time.perf_counter()
score = DecisionTreeClassifier(random_state=0).fit(X, y).score(X, y)
print(f"最小例子：鸢尾花 {len(X)} 条，训练准确率 {score:.2f}，用时 {time.perf_counter() - start:.2f} 秒")
print("\n环境检查通过，可以开始做实验了。")
