# Softsensor Library(softsensor-lib)
---
## 项目简介
softsensor-lib是一个面向深度学习研究者的开源库，特别适用于软测量研究分析

当前版本 v1

版本 softsensor-lib v1

🚩[最新动态] (2026.02) 我们新增了CNN和TCN模型

🚩[最新动态] (2026.02) 我们完成了针对软测量多目标值预测的实验框架

版本 softsensor-lib v0

🚩[最新动态] (2026.01) 我们完成了针对软测量多步预测的实验框架

🚩[最新动态] (2026.01) 我们完成了针对软测量单步预测的实验框架

🚩[最新动态] (2026.01) 我们开始设计一个适用于软测量任务的实验框架

🐎[最新动态]（2026.01）我们收集了部分开源的软测量数据集

---

## 快速开始

### 1. 准备数据

将软测量数据置于 ```./dataset``` 目录

### 2. 安装

1. 克隆本仓库

2. 创建新的 Conda 环境

3. 安装核心依赖

### 3. 训练与评估

1. 运行```./scripts```脚本文件

### 4. 开发自定义模型

1. 将模型文件放入 ```./models```, 可参考 ```./models/GRU```

2. 在```./models/__init__.py```中导入模型

3. 在```./exp/exp_basic.py```的```Exp_Basic.model_dict```中注册新模型。

4. 在```./scripts```下创建对应的运行脚本

## 引用

如果本仓库对您有帮助，请给我点个⭐，您的⭐是对我们最大的鼓励

如果要在您的论文中引用此仓库请参考 [link](https://www.wikihow.com/Cite-a-GitHub-Repository)

## 联系方式

如有问题或建议，欢迎联系维护团队

jitangjin@mail.sdu.edu.cn

## 致谢

本库参考了以下仓库：

[TSlib](https://github.com/thuml/Time-Series-Library)

实验所用数据集均为公开数据

[Link](https://link.springer.com/book/10.1007/978-1-84628-480-9)

