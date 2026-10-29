# 机器学习

面向成人教育人工智能技术应用专业（专科）的《机器学习》网络教材，从系列模板 `ai-textbook-template` 生成。

- 5 学分，90 学时（理论 60 + 实验 30），第 3 学期，期末闭卷
- 12 章，每章 3 个实验，共 36 个；神经网络章节给 PyTorch 和 Keras 两份代码
- 对应证书：人工智能训练师（三级）、人工智能数据处理工程师（高级）、Python 技术应用工程师（高级）、人工智能应用工程师（高级），见 `docs/certificate.md`
- 章节大纲见 `大纲-机器学习-v1.md`；动笔前先读 `编写规范.md`

## 目录结构

| 路径 | 内容 |
|---|---|
| `docs/chapters/` | 各章正文，目录名 `NN-英文短名` |
| `docs/sims/` | 交互动画，公共样式在 `common/` |
| `docs/learning-graph/` | 学习图谱（`data.js`）；`concepts-draft.csv` 是大纲阶段整理的全书概念清单草稿，由 `tools/build_concepts.py` 生成 |
| `labs/` | 按章存放的实验代码；`labs/data/` 为数据，`make_datasets.py` 生成模拟数据 |
| `tools/` | 维护脚本，用法见 `编写规范.md` 第十一节 |

## 本地预览

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

推送到 `main` 分支后，GitHub Actions 自动发布到 GitHub Pages（仓库 Settings → Pages → Source 选 GitHub Actions）。

## 说明

书中运行结果均为编者实测；需要 GPU、联网下载模型的部分标为“待实测”。模拟数据由 `labs/data/make_datasets.py` 按固定随机种子生成，内容为虚构，仅用于教学；数据来源与许可见 `labs/data/README.md`。
