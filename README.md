# ECG Heartbeat Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This repository implements a Deep Convolutional Neural Network (1-D ResNet) to automatically classify electrocardiogram (ECG) heartbeats into five distinct categories. It provides a full pipeline from raw signal processing to model training, evaluation, and experimental comparison, with a primary focus on addressing **class imbalance** in medical datasets using **SMOTE** (Synthetic Minority Over-sampling Technique).

## 🚀 Key Features

- **1-D Residual CNN**: A deep architecture with skip connections, mitigating vanishing gradients and learning robust spatial features from time-series signals.
- **Handling Imbalance**: Implements SMOTE on the training set to significantly improve the detection rate (recall) of rare, life-threatening arrhythmias (e.g., Supraventricular and Fusion beats).
- **Controlled Experimentation**: Contains a head-to-head comparison between a baseline model and a SMOTE-balanced model using the exact same neural network architecture and test set.
- **Reproducible Pipeline**: Modular codebase (`src/data.py`, `src/model.py`) allowing for easy modification, extension, and deployment.

## 📊 Quick Facts

| Feature | Details |
|---|---|
| **Domain** | Biomedical Engineering / Health Informatics |
| **Dataset** | PhysioNet MIT-BIH Arrhythmia Database |
| **Model Architecture** | 1-D Residual Convolutional Neural Network (1-D ResNet) |
| **Number of Classes** | 5 (AAMI EC57 Standard: N, S, V, F, Q) |
| **Testing Samples** | 21,892 |
| **Final Baseline Accuracy** | 98.70% |
| **Final SMOTE Accuracy** | 98.69% |
| **SMOTE Impact** | Improved minority class recall (S: +4.67%, F: +4.32%) |

## 📖 Comprehensive Documentation

This repository is heavily documented to support onboarding and research:
- **[Project Full Context](project_full_context.md)**: A complete, 95% self-contained technical knowledge base containing reading orders, codebase explanations, rationales, and deep dives.
- **[Theory Handbook](ECG_Heartbeat_Classification_Theory_Handbook.pdf)**: A beginner-friendly, 32-page illustrated PDF guide meant for viva preparation, presentation support, and concept review.
- **[Presentation Slides](ECG_Presentation_Final.pptx)**: An IEEE-style professional presentation deck summarizing the findings.

## 📁 Repository Structure

```text
ECG_Project/
├── data/                      # Dataset folder (CSV files are ignored by git)
│   └── README.md              # Instructions for downloading data
├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── Person_A_Baseline.ipynb 
│   └── Person_B_SMOTE.ipynb
├── results/                   # Experimental evidence
│   ├── figures/               # Generated graphs and confusion matrices
│   ├── models/                # Saved .keras model binaries
│   └── metrics_*.json         # Evaluation logs
├── src/                       # Core source code
│   ├── data.py                # Data loading and reshaping logic
│   └── model.py               # 1-D ResNet architecture definition
├── project_full_context.md    # Complete technical documentation
├── ECG_Heartbeat_Classification_Theory_Handbook.pdf
├── ECG_Presentation_Final.pptx
└── README.md
```

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ecg-heartbeat-classification.git
   cd ecg-heartbeat-classification
   ```

2. **Download the Dataset:**
   The MIT-BIH dataset is not included in this repository due to size constraints. Download the CSV files from Kaggle (`shayanfazeli/heartbeat`) and place `mitbih_train.csv` and `mitbih_test.csv` inside the `data/` folder. See [data/README.md](data/README.md) for details.

3. **Install Dependencies:**
   ```bash
   pip install tensorflow pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```

4. **Run the Notebooks:**
   Open and execute the Jupyter Notebooks in the `notebooks/` directory sequentially, or run the modules directly.

## 👥 Credits

- **Ribbie Mohammad Omar** (Roll: 2110041)
- **Md. Abdullah Ibna Shad** (Roll: 2110042)

Department of Biomedical Engineering.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
