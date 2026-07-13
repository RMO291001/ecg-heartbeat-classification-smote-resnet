# Project Full Context: ECG Heartbeat Classification

This document serves as the comprehensive knowledge base, technical documentation, architecture guide, experiment log, onboarding guide, and long-term context repository for the ECG Heartbeat Classification project. It is intended for future developers, research students, supervisors, and team members who will continue or review this work.

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Recommended Reading Order](#2-recommended-reading-order)
3. [Research Background](#3-research-background)
4. [Project Objectives](#4-project-objectives)
5. [Repository Structure](#5-repository-structure)
6. [Project Workflow](#6-project-workflow)
7. [Project Dependency Graph](#7-project-dependency-graph)
8. [Dataset Documentation](#8-dataset-documentation)
9. [Data Processing Pipeline](#9-data-processing-pipeline)
10. [Source Code Documentation](#10-source-code-documentation)
11. [Model Architecture](#11-model-architecture)
12. [Training Pipeline](#12-training-pipeline)
13. [Design Decisions](#13-design-decisions)
14. [Baseline Experiment](#14-baseline-experiment)
15. [SMOTE Experiment](#15-smote-experiment)
16. [Results Folder Documentation](#16-results-folder-documentation)
17. [Experimental Results](#17-experimental-results)
18. [Discussion](#18-discussion)
19. [Known Limitations](#19-known-limitations)
20. [Conclusions](#20-conclusions)
21. [Extending the Project](#21-extending-the-project)
22. [Future Improvements](#22-future-improvements)
23. [Developer Context](#23-developer-context)
24. [Dependencies](#24-dependencies)
25. [Reproduction Guide](#25-reproduction-guide)
26. [Developer Notes](#26-developer-notes)
27. [Credits](#27-credits)
28. [Appendix](#28-appendix)

---

## 1. Project Overview

### What this project does
This project implements a Deep Convolutional Neural Network (1-D ResNet) to automatically classify electrocardiogram (ECG) heartbeats into five distinct categories. It provides a full pipeline from raw signal processing to model training, evaluation, and experimental comparison.

### Quick Facts

| Feature | Details |
|---|---|
| **Project Name** | ECG Heartbeat Classification |
| **Project Type** | Academic Research / Open Source |
| **Domain** | Biomedical Engineering / Health Informatics |
| **Machine Learning Task** | Multi-class Time-series Classification |
| **Dataset** | PhysioNet MIT-BIH Arrhythmia Database |
| **Dataset Source** | Kaggle (`shayanfazeli/heartbeat`) |
| **Framework** | TensorFlow / Keras |
| **Programming Language** | Python |
| **Model Architecture** | 1-D Residual Convolutional Neural Network (1-D ResNet) |
| **Number of Classes** | 5 (AAMI EC57 Standard) |
| **Training Samples** | 87,554 |
| **Testing Samples** | 21,892 |
| **Final Baseline Accuracy** | 98.70% |
| **Final SMOTE Accuracy** | 98.69% |
| **Primary Goal** | Improve minority class recall in imbalanced ECG data using SMOTE |
| **Team Members** | Ribbie Mohammad Omar, Md. Abdullah Ibna Shad |

### Why it exists
Manual analysis of continuous ECG signals is extensively time-consuming and highly prone to human error. Cardiologists must sift through thousands of beats to find rare anomalies. This project automates the detection process, serving as a reliable digital assistant for cardiovascular monitoring.

### The problem it solves
The primary problem in automated heartbeat classification is **class imbalance**. In real-world data (and in the MIT-BIH dataset), normal heartbeats overwhelmingly outnumber abnormal ones (arrhythmias). Standard machine learning models often optimize for overall accuracy by heavily biasing towards the majority class, resulting in poor sensitivity (recall) for rare and dangerous arrhythmias.

### The motivation
The motivation is to take a state-of-the-art baseline architecture (a deep transferable representation proposed by Kachuee et al., 2018) and systematically address its weakness regarding class imbalance. By introducing Synthetic Minority Over-sampling Technique (SMOTE), the project aims to force the network to learn the minority class boundaries more effectively.

### Expected outcomes
The expected outcome is a robust, reproducible pipeline that demonstrates how SMOTE can improve the recall (sensitivity) of minority arrhythmia classes (like Supraventricular and Fusion beats) without significantly degrading the overwhelming baseline accuracy provided by the Normal beats.

---

## 2. Recommended Reading Order

If you are...

### New developer
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#5-repository-structure)
3. [Project Workflow](#6-project-workflow)
4. [Source Code Documentation](#10-source-code-documentation)
5. [Reproduction Guide](#25-reproduction-guide)

### Researcher
1. [Dataset Documentation](#8-dataset-documentation)
2. [Model Architecture](#11-model-architecture)
3. [Experimental Results](#17-experimental-results)
4. [Discussion](#18-discussion)

### Maintainer
1. [Repository Structure](#5-repository-structure)
2. [Project Dependency Graph](#7-project-dependency-graph)
3. [Developer Notes](#26-developer-notes)
4. [Extending the Project](#21-extending-the-project)

---

## 3. Research Background

### ECG Heartbeat Classification
An electrocardiogram (ECG) records the electrical activity of the heart over time. Heartbeat classification involves segmenting this continuous time-series signal into individual beats (usually centered around the R-peak, the most prominent spike) and categorizing the morphology of that beat into predefined medical classes.

### Arrhythmia Detection
Arrhythmias are irregular heartbeats. They can be premature (firing too early), abnormally shaped, or originating from the wrong part of the heart (e.g., ventricular instead of atrial). Accurate detection is critical because certain arrhythmias can precipitate severe cardiovascular events, including stroke or sudden cardiac death.

### MIT-BIH Dataset
The PhysioNet MIT-BIH Arrhythmia Database is the gold standard dataset for this task. It contains annotated ECG recordings from 47 subjects. The dataset has been preprocessed and segmented into 187-sample windows at 125 Hz, mapped into five Association for the Advancement of Medical Instrumentation (AAMI) EC57 standard categories.

### Why Class Imbalance is a Problem
In the MIT-BIH dataset, approximately 82.7% of the beats are Normal (Class N). If a classifier simply predicts "Normal" for every beat, it achieves ~82.7% accuracy while completely failing its medical purpose. We need high sensitivity to the minority classes, which standard loss functions and class distributions do not naturally provide.

### Why CNNs Work Well
Convolutional Neural Networks (CNNs), specifically 1-D CNNs, excel at this task because they automatically learn spatial (temporal) hierarchies of features. Historically, researchers used complex, handcrafted statistical features and wavelets. Deep CNNs replace this by learning the optimal filter weights to detect morphological anomalies (like a widened QRS complex or absent P-wave) directly from the raw signal.

### Why SMOTE was Investigated
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic examples of minority classes by interpolating between existing minority samples in the feature space. It was investigated because it prevents the network from simply memorizing duplicated minority samples (as in naive random oversampling) and expands the decision boundary for rare classes.

---

## 4. Project Objectives

### Technical Objectives
1. **Reproduce State-of-the-Art Baseline:** Successfully implement the 1-D Residual CNN architecture described by Kachuee et al. using TensorFlow/Keras.
2. **Implement Data Pipelines:** Build robust loading, reshaping, and preprocessing pipelines for the MIT-BIH CSV dataset.
3. **Integrate SMOTE:** Apply SMOTE correctly to the flattened temporal data exclusively within the training set, preventing any data leakage into the test set.
4. **Develop Evaluation Framework:** Create comprehensive evaluation scripts that output confusion matrices, per-class precision/recall/F1 metrics, and comparative plots.

### Research Objectives
1. **Analyze Imbalance Impact:** Quantify the baseline model's performance decay on minority classes (S, F, V) despite high overall accuracy.
2. **Evaluate SMOTE Efficacy:** Determine if spatial interpolation (SMOTE) on raw time-series ECG data provides meaningful clinical improvement in recall for rare arrhythmias.
3. **Examine Trade-offs:** Investigate the precision-recall trade-off that occurs when synthetically inflating minority distributions.

---

## 5. Repository Structure

### Repository Tree Diagram

The project structure is carefully organized to separate raw data, experimentation, reusable source code, and empirical results.

```text
ECG_Project/
│
├── data/
│   ├── mitbih_test.csv            # Test set (21,892 samples)
│   └── mitbih_train.csv           # Training set (87,554 samples)
│
├── notebooks/
│   ├── .ipynb_checkpoints/        # Jupyter auto-saves
│   ├── Person_A_Baseline.ipynb    # Scratchpad for baseline implementation and EDA
│   └── Person_B_SMOTE.ipynb       # Execution notebook for SMOTE application and evaluation
│
├── results/
│   ├── figures/                   # Auto-generated plots
│   │   ├── class_distribution.png # EDA: Class counts log-scale
│   │   ├── confusion_baseline.png # Test set confusion matrix (Baseline)
│   │   ├── confusion_side_by_side.png # Visual comparison
│   │   ├── confusion_smote.png    # Test set confusion matrix (SMOTE)
│   │   ├── f1_comparison.png      # Bar chart of F1 scores
│   │   └── sample_beats.png       # Raw ECG waveform samples per class
│   ├── models/                    # Saved Keras model weights
│   │   ├── baseline.keras
│   │   └── smote.keras
│   ├── comparison_table.csv       # Side-by-side metric deltas
│   ├── metrics_baseline.json      # Raw classification report (Baseline)
│   ├── metrics_smote.json         # Raw classification report (SMOTE)
│   └── summary.md                 # Brief textual summary of findings
│
├── src/
│   ├── __pycache__/
│   ├── data.py                    # Data loading and reshaping functions
│   └── model.py                   # 1-D ResNet architecture definitions
│
├── ECG_Heartbeat_Classificat_on_A_Deep_Transferable_Representation.md # Background Literature
├── ECG_Classification.ipynb       # Master project notebook
├── ECG_Heartbeat_Classification.pptx # Slide deck presentation
├── plan.md                        # Original execution roadmap and strategy
└── project_full_context.md        # This document
```

**Importance of Repository Structure**: This directory layout is standard for reproducible machine learning research. It ensures data isn't accidentally pushed to version control, notebooks have a clean execution space, and results are safely archived alongside the exact models that generated them.

---

## 6. Project Workflow

### Project Workflow Diagram

The logical sequence of operations from data ingestion to final reporting.

```text
[Raw Dataset (Kaggle)]
         ↓
[Exploratory Data Analysis (EDA)] --> (Class Distribution, Waveform Checking)
         ↓
[Preprocessing] --> (Pandas CSV Load -> Numpy Array -> Reshape (N, 187, 1))
         ↓
==================================================
|                   BRANCHING                    |
==================================================
      ↓                                    ↓
[Baseline Path]                      [SMOTE Path]
      ↓                                    ↓
(Use raw Train Set)         (Flatten -> SMOTE -> Reshape Train Set)
      ↓                                    ↓
[Model Training]                     [Model Training]
(Compile ResNet, Adam, Epochs=30, EarlyStopping)
      ↓                                    ↓
[Evaluation]                         [Evaluation]
(Test Set Inference, Confusion Matrix, Classification Report)
      ↓                                    ↓
==================================================
|                   MERGING                      |
==================================================
         ↓
[Comparison & Analysis] --> (F1, Recall, Precision Deltas, Side-by-Side Plots)
         ↓
[Presentation / Reporting]
```

**Importance of Project Workflow**: The branching execution path clearly illustrates how the baseline acts as a strict control group for the SMOTE experiment, ensuring an unbiased comparison.

---

## 7. Project Dependency Graph

### Dependency Graph Diagram

This diagram shows how code files and assets rely on one another.

```text
[Raw Dataset: mitbih_train.csv / mitbih_test.csv]
    ↓
src/data.py
    ↓
src/model.py
    ↓
[Model Training: baseline.keras / smote.keras]
    ↓
[Evaluation Metrics: metrics_baseline.json / metrics_smote.json]
    ↓
[Comparative Analysis: comparison_table.csv]
    ↓
[Visualizations: f1_comparison.png, confusion_side_by_side.png]
    ↓
[Final Reporting: project_full_context.md, summary.md, Presentation]
```

**Importance of Dependency Graph**: Understanding this graph is essential for pipeline debugging. If a metric changes, developers can trace backward to identify exactly which stage (e.g., `data.py` or `model.py`) introduced the modification.

---

## 8. Dataset Documentation

### Source & Properties
- **Source**: PhysioNet MIT-BIH Arrhythmia Database (Kaggle: `shayanfazeli/heartbeat`)
- **Train File**: `mitbih_train.csv` (87,554 samples)
- **Test File**: `mitbih_test.csv` (21,892 samples)
- **Total Classes**: 5 (AAMI EC57 standard)
- **ECG Signal Length**: 187 time-steps (representing ~1.5 seconds at 125 Hz).
- **Format**: CSV where columns 0-186 are normalized amplitude values [0, 1], and column 187 is the integer class label.

### Dataset Statistics Table

| Class ID | Abbreviation | Medical Meaning | Count | Percentage |
|---|---|---|---|---|
| 0 | N | Normal beat, Left/Right bundle branch block, Atrial/Nodal escape | 72,471 | ~82.7% |
| 1 | S | Supraventricular premature beat (Atrial, Nodal) | 2,223 | ~2.5% |
| 2 | V | Premature ventricular contraction, Ventricular escape | 5,788 | ~6.6% |
| 3 | F | Fusion of ventricular and normal beat | 641 | ~0.7% |
| 4 | Q | Paced beat, Fusion of paced and normal, Unclassifiable | 6,431 | ~7.3% |

**Importance of Dataset Statistics**: It quantitatively highlights the extreme class imbalance that motivates the entire SMOTE experiment.

### Class Distribution Plot

This chart visually illustrates the skewed nature of the dataset.

![Class Distribution Plot](results/figures/class_distribution.png)

**Caption:** MIT-BIH Training Set Class Distribution (Log Scale).  
**Importance:** Confirms the heavy dominance of Class N, visualizing the disparity between normal beats and arrhythmias.

### Sample ECG Beats Plot

Visualizing one sample from each class helps verify the biological nature of the signal.

![Sample ECG Beats](results/figures/sample_beats.png)

**Caption:** Sample waveform for each of the 5 AAMI classes.  
**Importance:** Proves the raw CSV data correlates with actual ECG morphologies, serving as a critical sanity check for data integrity.

---

## 9. Data Processing Pipeline

### Preprocessing Pipeline Diagram

```text
[Raw CSV File] 
      ↓ (Pandas)
[Dataframe (N rows x 188 cols)]
      ↓ (Split)
[Features (0-186)]   [Labels (187)]
      ↓ (Numpy)         ↓ (Int)
[Array (N, 187)]     [Array (N, 1)]
```

### Tensor Shape Transformation Diagram

```text
Baseline Pipeline:
[Array (N, 187)] -> reshape(-1, 187, 1) -> [Tensor (N, 187, 1)] -> (To Model)

SMOTE Pipeline:
[Array (N, 187)] -> (Flattened internally) -> SMOTE() 
    -> [Balanced Array (M, 187)] -> reshape(-1, 187, 1) -> [Balanced Tensor (M, 187, 1)] -> (To Model)
```

**Importance of Shape Transformation**: Convolutional layers require a "channel" dimension. Reshaping from `(N, 187)` to `(N, 187, 1)` is a mandatory step that frequently trips up new developers.

---

## 10. Source Code Documentation

Important snippets from the codebase that drive the logic.

### Data Loader Function (`load_and_prepare()`)

This function ingests and transforms the raw dataset.

```python
def load_and_prepare(train_path="data/mitbih_train.csv", test_path="data/mitbih_test.csv"):
    train = pd.read_csv(train_path, header=None)
    test  = pd.read_csv(test_path, header=None)
    
    # Reshape features for 1D CNN: (samples, time_steps, channels)
    X_train = train.iloc[:, :187].values.reshape(-1, 187, 1)
    y_train = train.iloc[:, 187].values.astype(int)
    
    X_test  = test.iloc[:, :187].values.reshape(-1, 187, 1)
    y_test  = test.iloc[:, 187].values.astype(int)
    
    return X_train, y_train, X_test, y_test
```
**Importance**: This is the universal starting point for any script or notebook in the project.

### SMOTE Implementation

The script snippet used to synthesize minority instances.

```python
from imblearn.over_sampling import SMOTE

# SMOTE requires 2D input, so we flatten the 187x1 arrays
X_train_flat = X_train.reshape(len(X_train), -1)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_bal_flat, y_bal = smote.fit_resample(X_train_flat, y_train)

# Reshape back to 3D for the CNN: (samples, time_steps, channels)
X_bal = X_bal_flat.reshape(-1, 187, 1)
```
**Importance**: The central contribution of the project. Note the necessary flattening and reshaping required to bridge the gap between `imblearn` and Keras.

---

## 11. Model Architecture

The core inference engine is a 1-D Convolutional Neural Network heavily inspired by ResNet architectures.

### Architecture Diagram

```text
[ Input: (187, 1) ]
       ↓
[ Conv1D (32, 5) + ReLU ]
       ↓
[ Residual Block 1 ] ---> (Conv -> ReLU -> Conv -> Add -> ReLU -> MaxPool)
       ↓
[ Residual Block 2 ]
       ↓
[ Residual Block 3 ]
       ↓
[ Residual Block 4 ]
       ↓
[ Residual Block 5 ]
       ↓
[ Flatten ]
       ↓
[ Dense (32) + ReLU ]
       ↓
[ Dense (32) + ReLU ]
       ↓
[ Dense (5) + Softmax ] ---> [ Output Probability ]
```

### Residual Block Architecture

The code implementing the skip connections inside each block.

```python
def residual_block(x, filters=32, kernel_size=5):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(pool_size=5, strides=2, padding="same")(x)
    return x
```
**Importance**: Skip connections mitigate the vanishing gradient problem, enabling the network to grow 13 layers deep while maintaining effective training.

### Full Network Construction (`build_kachuee_cnn()`)

```python
def build_kachuee_cnn(input_shape=(187, 1), n_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, padding="same")(inputs)
    x = layers.ReLU()(x)
    for _ in range(5):
        x = residual_block(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)
```

### Model Statistics

- **Input Shape**: `(187, 1)`
- **Output Shape**: `(5,)` (Probability distribution over 5 classes)
- **Number of Residual Blocks**: 5
- **Number of Convolutional Layers**: 11 (1 initial + 2 per residual block)
- **Number of Dense Layers**: 3 (2 hidden, 1 output)

### Model Summary Output

> [MODEL SUMMARY PLACEHOLDER]

**Importance of Model Summary**: It provides a layer-by-layer breakdown of feature map dimensions and parameter counts, crucial for profiling memory usage.

---

## 12. Training Pipeline

### Model Compilation and Fitting (`model.compile()` & `model.fit()`)

```python
model_smote.compile(optimizer=Adam(1e-3),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

history_smote = model_smote.fit(
    X_bal, y_bal,
    validation_split=0.1,
    epochs=30,
    batch_size=256,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
               ReduceLROnPlateau(patience=3, factor=0.5)]
)
```
**Importance**: This snippet dictates exactly how the network learns, defining the loss landscape traversal methodology.

### Hyperparameter Configuration

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam |
| **Learning Rate** | 1e-3 (0.001) |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Batch Size** | 256 |
| **Epochs** | 30 |
| **Validation Split** | 0.1 (10%) |
| **EarlyStopping Patience** | 5 |
| **ReduceLROnPlateau Patience** | 3 |
| **Learning Rate Factor** | 0.5 |
| **Input Shape** | (187, 1) |
| **Output Classes** | 5 |

### Training Curves

> [TRAINING CURVE PLACEHOLDER]

**Caption:** Training vs Validation Accuracy and Loss Curves.
**Importance:** Plotting the training curves verifies that the model does not suffer from extreme variance (overfitting) and demonstrates the points at which `EarlyStopping` naturally terminated the session.

---

## 13. Design Decisions

- **Conv1D instead of 2D CNN**: ECG signals are inherently one-dimensional time-series data. 1-D convolutions naturally capture temporal morphology (such as QRS complex widths) without the unnecessary computational overhead of a 2D spatial transformation (e.g., converting to spectrograms).
- **Residual CNN instead of a shallow CNN**: Deep networks often suffer from the vanishing gradient problem. Residual connections allow gradients to flow directly through the network, enabling the training of deeper architectures that capture more complex, abstract hierarchical features.
- **Adam optimizer**: Selected for its adaptive moment estimation, which provides rapid convergence and robustness to saddle points compared to standard Stochastic Gradient Descent.
- **Sparse Categorical Crossentropy**: Chosen to maintain label efficiency. It performs the same operation as categorical crossentropy but natively handles integer-encoded labels, reducing memory footprint.
- **EarlyStopping**: Ensures the model is evaluated at its optimal generalization point, rather than arbitrarily at the final epoch, inherently mitigating overfitting.
- **ReduceLROnPlateau**: Complements EarlyStopping by actively modulating the learning rate, allowing the optimizer to traverse narrow loss valleys effectively.
- **Batch Size = 256**: Balances computational throughput on GPU architectures with gradient stability, which is particularly critical given the dataset's skew.
- **Validation Split = 10%**: Reserves a sufficient number of samples (~8,755) to provide a statistically significant validation signal without excessively depleting the training corpus.
- **SMOTE instead of Random Oversampling**: Random oversampling duplicates exact minority samples, which strongly predisposes the network to overfit on memorized instances. SMOTE synthesizes novel samples, encouraging the network to generalize the minority class boundaries.
- **Same architecture for Baseline and SMOTE**: Maintaining architectural equivalence isolates the data balancing technique as the sole independent variable, ensuring the experimental comparison remains scientifically valid and fair.

---

## 14. Baseline Experiment

### Goal
Establish a control metric by training the CNN on the raw, imbalanced MIT-BIH dataset to replicate the ~93.4%+ accuracy claims of prior literature.

### Procedure
The data is loaded, reshaped, and fed directly into the model. No augmentation is applied. 

### Expected vs Actual Behaviour
Expected: High overall accuracy, but poor minority recall.
Actual: The model achieved an overall test accuracy of 98.70%, substantially exceeding the initial performance target. However, closer inspection reveals the bias: Class N precision/recall are near perfect, but Class S (Supraventricular) recall is only **79.3%**, and Class F (Fusion) is only **78.4%**. The model is missing over 20% of these critical arrhythmias.

### Performance Comparison to Prior Work
The model achieved an overall test accuracy of 98.70%, which substantially exceeds the ~93.4% accuracy reported in the reference literature by Kachuee et al. (2018). While a direct exact replication is challenging, this variance can plausibly be attributed to:
- **Experimental Configuration and Hyperparameters**: Differences in early stopping patience, learning rate schedules, or exact validation splits.
- **Evaluation Protocol**: Potential discrepancies in how the test set was partitioned or sampled.
- **Implementation Details**: Subtle differences in padding strategies, initialization techniques, or framework-specific optimizer behaviors between the original computational environments and modern Keras implementations.

---

## 15. SMOTE Experiment

### Why SMOTE?
To fix the 20% miss rate on rare classes, we must balance the training data. However, simple random oversampling (copy-pasting rare beats) causes neural networks to severely overfit to those specific copies. SMOTE (Synthetic Minority Over-sampling Technique) creates *new, synthetic* waveforms by drawing lines between existing minority samples in the 187-dimensional space and picking points along those lines.

### Oversampling Training Data ONLY
It is absolutely vital that SMOTE is applied *only* to the training set. If synthetic data leaks into the test set, the evaluation is corrupted, leading to artificially inflated, invalid accuracy scores.

### Comparison
The SMOTE model trained on equal distributions of N, S, V, F, and Q. It was then evaluated on the exact same, highly imbalanced real-world test set as the baseline.

---

## 16. Results Folder Documentation

The `results/` folder contains all empirical evidence.

### `metrics_baseline.json` & `metrics_smote.json`
- **Purpose**: To persistently store the precise classification metrics (precision, recall, F1-score, support) for each class.
- **Importance**: Acts as the immutable quantitative record of model performance.

> [JSON CONTENT PLACEHOLDER — metrics_baseline.json excerpt]
> [JSON CONTENT PLACEHOLDER — metrics_smote.json excerpt]

### `comparison_table.csv`
- **Purpose**: To provide a direct, side-by-side mathematical comparison of the baseline and SMOTE performances.
- **Importance**: Crucial for quantifying the exact clinical trade-offs induced by SMOTE.

> [CSV CONTENT PLACEHOLDER — comparison_table.csv]

### `summary.md`
- **Purpose**: To distill the complex mathematical results into an accessible executive summary.
- **Importance**: Serves as the primary communicative artifact of the project's clinical conclusions.

> [MARKDOWN CONTENT PLACEHOLDER — summary.md]

### `baseline.keras` & `smote.keras`
- **Purpose**: To serialize the trained neural network architectures and their learned optimal weights.
- **Importance**: Represent the final deliverable computational assets of the training phase.

> [MODEL BINARY PLACEHOLDER — baseline.keras]
> [MODEL BINARY PLACEHOLDER — smote.keras]

---

## 17. Experimental Results

### Baseline Confusion Matrix

![Baseline Confusion Matrix](results/figures/confusion_baseline.png)

**Caption:** Confusion matrix for the baseline model (no balancing).  
**Importance:** Graphically identifies exactly where the model struggles (high off-diagonal counts for minority classes).

### SMOTE Confusion Matrix

![SMOTE Confusion Matrix](results/figures/confusion_smote.png)

**Caption:** Confusion matrix for the SMOTE-trained model.  
**Importance:** Shows the shifted distributions; more minority arrhythmias are correctly identified on the diagonal.

### Side-by-Side Confusion Matrix

![Side-by-Side Confusion](results/figures/confusion_side_by_side.png)

**Caption:** Side-by-side comparison of baseline and SMOTE matrices.  
**Importance:** The immediate visual proof of SMOTE's impact on classifying S, V, and F arrhythmias.

### F1 Comparison Chart

![F1 Comparison](results/figures/f1_comparison.png)

**Caption:** Bar chart comparing baseline F1 scores against SMOTE F1 scores per class.  
**Importance:** Translates complex abstract metrics into an easily understandable presentation format for stakeholders.

### Evaluation Pipeline Code

The script responsible for generating the predictions and evaluating them.

```python
# Generate predictions
y_pred_smote = model_smote.predict(X_test).argmax(axis=1)
report_smote = classification_report(y_test, y_pred_smote, target_names=["N","S","V","F","Q"], output_dict=True)

# Plot Confusion Matrix
cm_smote = confusion_matrix(y_test, y_pred_smote)
sns.heatmap(cm_smote, annot=True, fmt="d", cmap="Greens",
            xticklabels=["N","S","V","F","Q"], yticklabels=["N","S","V","F","Q"])
```
**Importance:** Encapsulates the standard evaluation loop used to parse the softmax outputs into definitive class metrics.

### Final Metrics Summary Table

| Metric | Baseline | SMOTE | Delta |
|---|---|---|---|
| **Overall Accuracy** | 98.70% | 98.69% | -0.01% |
| **Macro F1** | 92.78% | 92.81% | +0.03% |
| **Weighted F1** | 98.66% | 98.67% | +0.01% |

### Per-Class Analysis
- **Class N**: Recall shifted from 99.7% to 99.4%. Precision stayed ~99%.
- **Class S**: Recall **improved** from 79.3% to **84.0%** (+4.67%).
- **Class V**: Recall shifted from 95.0% to 95.5% (+0.5%).
- **Class F**: Recall **improved** from 78.4% to **82.7%** (+4.32%).
- **Class Q**: Recall shifted from 98.8% to 99.1% (+0.3%).

---

## 18. Discussion

### The Precision-Recall Tradeoff
The results perfectly illustrate a classic machine learning trade-off. By using SMOTE, we shifted the decision boundaries. The network is now highly vigilant for arrhythmias. In a medical context, this is highly desirable: a false positive (calling a normal beat an arrhythmia) results in a doctor double-checking the chart. A false negative (missing an actual arrhythmia) could result in patient death. Therefore, trading precision for recall on minority classes represents a clinical success.

### Overall Accuracy Illusion
Overall accuracy remained totally flat (98.70% vs 98.69%). This is because Class N comprises 82% of the test set. A minor fractional drop in Class N accuracy completely masks the substantial 5% improvements in the rare classes. This finding corroborates the notion that accuracy is a fundamentally flawed metric in imbalanced medical machine learning.

---

## 19. Known Limitations

While this project demonstrates substantial success, it is constrained by several known limitations:

- **Single-lead ECG**: The model relies exclusively on a single ECG lead (Lead II). In clinical practice, 12-lead ECGs are utilized to provide a comprehensive spatial understanding of cardiac electrical activity.
- **MIT-BIH Dataset Only**: The model has been trained and tested exclusively on the MIT-BIH Arrhythmia Database. It has not been validated against external datasets to confirm generalizability.
- **No Patient-wise Cross-validation**: The standard train/test split does not strictly enforce patient isolation. It is possible that beats from the same patient exist in both sets, potentially inflating the reported accuracy.
- **SMOTE Physiological Plausibility**: SMOTE performs linear interpolation in high-dimensional vector space. It generates mathematically synthetic signals that may lack genuine physiological plausibility (e.g., creating artifactual waveforms that do not exist in human biology).
- **No Uncertainty Estimation**: The neural network outputs softmax probabilities but lacks a principled mechanism for uncertainty quantification (e.g., Bayesian inference), which is critical in medical diagnostics.
- **Single Architecture Evaluated**: The experiment relies solely on the 1-D ResNet architecture, without benchmarking against newer paradigms like Transformers or established baselines like Random Forests.
- **No Explainability Methods**: The project currently lacks post-hoc interpretability (e.g., Grad-CAM, SHAP, LIME) to identify which morphological segments of the ECG (like the P-wave or QRS complex) drive the classification decisions.
- **No Real-time Deployment Evaluation**: The inference latency and computational overhead have not been benchmarked for deployment on low-power, real-time continuous monitoring edge devices.

---

## 20. Conclusions

- **Successes**: The project successfully replicated a highly complex deep learning baseline and effectively utilized synthetic oversampling to address a critical domain problem. Minority class sensitivity was meaningfully increased.
- **Final Outcome**: The SMOTE model proves to be medically superior to the baseline model despite exhibiting a marginally lower mathematical accuracy, primarily due to its significantly enhanced recall on dangerous arrhythmias.

---

## 21. Extending the Project

Future developers are encouraged to iterate on this foundation. Here are actionable pathways for extension:

- **Replace the CNN with Another Architecture**: To test a Transformer or LSTM, modify the `build_kachuee_cnn()` function within `src/model.py`. Ensure the input shape `(187, 1)` is preserved.
- **Replace SMOTE with Class Weighting**: Bypass the SMOTE application in the training notebook. Instead, compute class weights using `sklearn.utils.class_weight` and pass the resulting dictionary to the `class_weight` argument in `model.fit()`.
- **Add Focal Loss**: Implement a custom Focal Loss function in TensorFlow/Keras and supply it to the `loss` parameter during `model.compile()` to natively handle the imbalance without oversampling.
- **Add Attention Mechanisms**: Augment the existing `residual_block` in `src/model.py` with spatial or temporal attention layers (e.g., Squeeze-and-Excitation blocks) to help the network focus on the QRS complex.
- **Add Transformers**: Implement a Vision Transformer (ViT) adapted for 1-D sequences. This will require modifying `src/model.py` to include positional encodings and multi-head self-attention layers.
- **Add GAN-generated ECG Signals**: Replace the `imblearn.SMOTE` step with a pre-trained Generative Adversarial Network that synthesizes physiologically constrained, class-conditioned ECG beats.
- **Add Cross-validation**: Refactor the data loading pipeline to support GroupKFold cross-validation based on patient IDs (if external metadata is acquired), ensuring robust out-of-sample evaluation.
- **Deploy the Trained Model**: Utilize the serialized `smote.keras` file to deploy the model via FastAPI or TensorFlow Serving for real-time web inference.

---

## 22. Future Improvements

Beyond the extensions listed above, long-term future improvements could explore:
1. **Focal Loss**: Modify the loss function directly (e.g., Lin et al.) to dynamically scale the gradient based on prediction confidence, forcing the network to focus on hard minority examples without synthesizing fake data.
2. **Weighted Cross Entropy**: Simply apply class weights during `model.fit()` to penalize minority class errors exponentially.
3. **GAN Augmentation**: Use Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs) to generate biologically valid synthetic ECG waveforms instead of linear SMOTE interpolation.
4. **Attention Mechanisms / Transformers**: Replace the 1-D CNN with a Transformer architecture that can learn long-range temporal dependencies and focus on specific morphological abnormalities (like the PR interval).
5. **Ensemble Learning**: Train distinct models on specific one-vs-all tasks and aggregate their predictions.

---

## 23. Developer Context

### Notebook Execution Order

To accurately reproduce the project states from scratch, notebooks must be executed sequentially:
1. `Person_A_Baseline.ipynb` (Establishes EDA, baseline metrics)
2. `Person_B_SMOTE.ipynb` (Applies oversampling, evaluates, compares to baseline)

### Important Terminal Commands

> [OUTPUT PLACEHOLDER — Training Log]

### Environment Setup

Developers must ensure the environment contains the necessary data science stack.

```bash
pip install tensorflow pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```
**Importance**: Ensures all computational graphs and data processing pipelines possess their required dependencies prior to runtime.

### Folder Dependency Diagram

```text
data/ --> src/data.py
src/data.py --> notebooks/
src/model.py --> notebooks/
notebooks/ --> results/models/
notebooks/ --> results/figures/
```
**Importance**: Clarifies directory responsibilities and the direction of data flow within the repository.

---

## 24. Dependencies

- `tensorflow / keras`: The core deep learning framework. GPU support highly recommended.
- `pandas`: Required for parsing massive CSV datasets efficiently.
- `numpy`: The foundation for all tensor math and reshaping.
- `scikit-learn`: Provides `classification_report`, `confusion_matrix`, and metric calculations.
- `imbalanced-learn`: Provides the `SMOTE` implementation.
- `matplotlib / seaborn`: Required for generating heatmaps, bar charts, and waveform plots.

---

## 25. Reproduction Guide

To reproduce this project from scratch:

1. **Environment Setup**: Use the command provided in Developer Context to install libraries.
2. **Download Dataset**: Download the MIT-BIH dataset from Kaggle (`shayanfazeli/heartbeat`) and place `mitbih_train.csv` and `mitbih_test.csv` inside the `data/` folder.
3. **Run Baseline**: Execute `notebooks/Person_A_Baseline.ipynb` entirely to generate the baseline model and metrics.
4. **Run SMOTE**: Execute `notebooks/Person_B_SMOTE.ipynb` to apply SMOTE, train the second model, and generate the comparative json metrics.
5. **Analyze**: Review the generated outputs in the `results/figures/` and `results/` directories.

---

## 26. Developer Notes

- **IMPORTANT ASSUMPTION**: The input data is assumed to be exactly 187 time steps long, pre-centered around the R-peak, and pre-normalized to [0,1]. Any data in the wild will need extreme preprocessing (Pan-Tompkins algorithm, filtering, segmentation) to match this format before inference.
- **NEVER MODIFY**: Do not apply SMOTE or any balancing technique to `X_test`. Doing so guarantees data leakage and ruins the scientific integrity of the results.
- **Coding Conventions**: Follow standard Python PEP-8. Neural network architectures should remain modular in `src/model.py` returning uncompiled Keras models.

---

## 27. Credits

### Project Team

- **Ribbie Mohammad Omar**
  Roll: 2110041
  Responsibilities:
  - Baseline CNN implementation
  - Project architecture
  - Baseline experiment
  - Comparison framework
  - Result visualization
  - Documentation
  - Presentation preparation

- **Md. Abdullah Ibna Shad**
  Roll: 2110042
  Responsibilities:
  - SMOTE implementation
  - Balanced dataset generation
  - SMOTE experiment
  - Comparative evaluation
  - Result verification

### Academic & Technical Credits

- **Original Architecture & Concept**: Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). *ECG Heartbeat Classification: A Deep Transferable Representation*. University of California, Los Angeles (UCLA).
- **Dataset Source**: PhysioNet (Goldberger et al.), MIT-BIH Arrhythmia Database. Hosted in preprocessed format by Shayan Fazeli on Kaggle.
- **Libraries**: Built utilizing open-source tools: TensorFlow, Scikit-Learn, Imbalanced-Learn.

---

# 28. Appendix

### Appendix A — Repository Tree
*(Reference Section 5 for the detailed project structure tree)*

### Appendix B — Workflow Diagram
*(Reference Section 6 for the branching execution pipeline)*

### Appendix C — Model Summary
> [MODEL SUMMARY PLACEHOLDER]

### Appendix D — Important Code Snippets
*(Reference Section 10 and 11 for implementations like `build_kachuee_cnn()`)*

### Appendix E — Dataset Statistics
*(Reference Section 8 for the specific class distributions and sample counts)*

### Appendix F — Training Curves
> [TRAINING CURVE PLACEHOLDER]

### Appendix G — Confusion Matrices
*(Reference Section 17 for full resolution Confusion Matrix visuals)*

### Appendix H — Comparison Tables
> [TABLE PLACEHOLDER — Comparison Table]

### Appendix I — Environment Information
Ensure execution environment conforms to Python 3.8+ with GPU hardware acceleration enabled for optimal iteration speed.

### Appendix J — Useful Commands
```bash
# Example for executing model training notebooks via CLI
jupyter nbconvert --to script notebooks/Person_A_Baseline.ipynb
python notebooks/Person_A_Baseline.py
```
