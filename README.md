# Softsensor Library (softsensor-lib)

softsensor-lib is an open-source library designed for deep learning researchers, focusing on soft sensor modeling and experimental analysis.

The project aims to provide a unified, extensible, and experiment-friendly framework for soft sensor tasks, enabling researchers to quickly perform model training, evaluation, and comparative experiments.

## Project Overview

In industrial process modeling and soft sensing research, researchers often repeatedly perform tasks such as data preprocessing, model development, training evaluation, and experiment management.

To address these needs, softsensor-lib is designed as a well-structured and easily extensible experimental platform to help users efficiently conduct the following tasks:

Single-step prediction for soft sensing

Multi-step prediction for soft sensing

Multi-target prediction for soft sensing

Unified training and evaluation for multiple deep learning models

Rapid integration of custom models and reproducible experiments

This project is particularly suitable for:

Researchers working in soft sensing and process industry modeling

## Version Updates

Current version: v1

softsensor-lib v1

🚩 [Latest Update] (2026.02) Completed the experimental framework for multi-target soft sensing prediction

🚩 [Latest Update] (2026.01) Completed the experimental framework for multi-step soft sensing prediction

🚩 [Latest Update] (2026.01) Completed the experimental framework for single-step soft sensing prediction

🚩 [Latest Update] (2026.01) Started designing an experimental framework for soft sensing tasks

🐎 [Latest Update] (2026.01) Collected and organized several open-source soft sensing datasets

## Quick Start
### 1. Prepare Data

Place the soft sensing datasets in the following directory:```./dataset```

### 2. Installation

Clone this repository

Create a new Conda environment

Install the required dependencies

### 3. Training and Evaluation

Run the scripts in:```./scripts```

### 4. Developing Custom Models

Place the model file in:

```./models```

You may refer to:

```./models/GRU```

Import the model in:

```./models/__init__.py```

Register the model in:

```Exp_Basic.model_dict``` 

located in:

```./exp/exp_basic.py```

Create a corresponding running script in:```./scripts```

## Citation

If you find this repository helpful, please consider giving it a ⭐ star. Your support is greatly appreciated.

If you would like to cite this repository in your paper, please refer to:[link](https://www.wikihow.com/Cite-a-GitHub-Repository)

## Contact

If you have any questions or suggestions, please feel free to contact the maintainers:

jitangjin@mail.sdu.edu.cn

lapluie124@mail.sdu.edu.cn

## Acknowledgements

This library was inspired by the following repository:

[TSlib](https://github.com/thuml/Time-Series-Library)

All datasets used in the experiments are publicly available datasets.

[Link](https://link.springer.com/book/10.1007/978-1-84628-480-9)






