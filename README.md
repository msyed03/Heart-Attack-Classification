# Heart Attack Risk Prediction Using Machine Learning and Clinical Biomarkers

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)

</p>

---

## Overview

Heart disease remains one of the leading causes of death worldwide and continues to place a significant burden on healthcare systems. According to the World Health Organization (WHO), cardiovascular diseases account for approximately **18 million deaths each year**, with heart attacks representing one of the most common and life-threatening conditions. Although advances in medical technology have improved diagnosis and treatment, identifying patients at high risk before a cardiac event occurs remains a challenging task.

Machine learning has become an increasingly valuable tool in healthcare because it can recognize complex relationships within large datasets that are often difficult to detect using conventional statistical methods. Rather than replacing healthcare professionals, predictive models can serve as clinical decision-support tools by assisting physicians in identifying high-risk patients, prioritizing medical evaluations, and supporting preventative care strategies.

This project investigates the use of supervised machine learning techniques to predict heart attack risk using routinely collected patient information, including demographic characteristics, vital signs, and cardiac biomarkers. Multiple machine learning algorithms were developed, optimized, and evaluated to determine which model produced the strongest predictive performance while maintaining good generalization on unseen data.

Instead of focusing on a single classifier, this project compares several fundamentally different learning algorithms, evaluates multiple sampling strategies for handling class imbalance, performs hyperparameter tuning for every model, analyzes feature importance, and discusses the strengths and limitations of each approach.

The goal is not only to maximize predictive accuracy but also to better understand how different machine learning models behave when applied to structured clinical data.

---

## Project Highlights

- Developed a complete supervised machine learning pipeline for heart attack risk prediction.
- Compared **five classification algorithms** using identical evaluation procedures.
- Investigated the effects of **class imbalance** using multiple sampling strategies.
- Performed hyperparameter optimization for every machine learning model.
- Evaluated performance using a comprehensive set of classification metrics.
- Analyzed feature importance to identify the most influential clinical variables.
- Compared model robustness, generalization performance, and overfitting tendencies.
- Achieved **98.5% classification accuracy** using ensemble learning methods.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Business Value and Significance](#business-value-and-significance)
- [Dataset](#dataset)
- [Dataset Exploration](#dataset-exploration)
- [Methodology](#methodology)
- [Data Preprocessing](#data-preprocessing)
- [Sampling Techniques](#sampling-techniques)
- [Train-Test Split](#train-test-split)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Results](#model-results)
- [Feature Importance](#feature-importance)
- [Model Comparison](#model-comparison)
- [Overfitting Analysis](#overfitting-analysis)
- [Conclusion](#conclusion)
- [References](#references)

---

# Project Motivation

Heart attacks rarely occur because of a single abnormal measurement. Instead, physicians evaluate numerous clinical indicators simultaneously—including blood pressure, heart rate, blood glucose, cardiac enzyme levels, age, and other patient characteristics—to estimate cardiovascular risk.

While experienced healthcare professionals can interpret these variables effectively, analyzing multiple biomarkers together becomes increasingly complex as patient volume grows. Machine learning offers an opportunity to assist this process by identifying relationships between clinical variables that may not be immediately apparent.

Predictive models can learn from historical patient data and recognize patterns associated with elevated cardiovascular risk. When used responsibly, these models can support physicians by providing an additional data-driven perspective during clinical decision-making.

This project explores whether supervised machine learning algorithms can accurately classify patients according to heart attack risk using commonly available clinical measurements.

---

# Problem Statement

Heart disease remains one of the leading causes of mortality worldwide, accounting for millions of deaths each year. Early identification of individuals at elevated risk is essential because timely medical intervention can significantly reduce the likelihood of severe cardiovascular events.

Traditional risk assessment often depends on the interpretation of multiple laboratory tests and physiological measurements. Although these indicators are well understood individually, combining them into a reliable prediction can be challenging.

The objective of this project is to develop and evaluate machine learning models capable of predicting whether a patient is at risk of experiencing a heart attack using demographic information, vital signs, and cardiac biomarkers.

The project investigates multiple supervised learning algorithms, compares different sampling strategies for handling class imbalance, and identifies the approach that provides the strongest predictive performance while maintaining reliable generalization.

---

# Project Objectives

The primary objectives of this project include:

- Develop a binary classification model capable of predicting heart attack risk.
- Explore and understand the characteristics of the clinical dataset.
- Perform data preprocessing to improve data quality and model performance.
- Investigate the impact of class imbalance on predictive accuracy.
- Compare multiple oversampling and undersampling strategies.
- Train and optimize several supervised machine learning algorithms.
- Evaluate every model using multiple classification metrics.
- Compare strengths and weaknesses of different learning approaches.
- Analyze feature importance to identify clinically significant variables.
- Determine which algorithm provides the best overall predictive performance.

---

# Business Value and Significance

Although this project focuses on machine learning research, its practical applications extend well beyond an academic setting.

Predictive models capable of identifying patients at elevated cardiovascular risk have the potential to improve healthcare delivery in several ways.

## Early Detection

Machine learning models can identify patients who may require additional medical evaluation before severe cardiovascular symptoms appear.

Early detection allows healthcare providers to intervene sooner, potentially reducing the likelihood of heart attacks through preventative treatment and lifestyle modifications.

## Clinical Decision Support

Rather than replacing physicians, predictive models can function as clinical decision-support tools.

By analyzing multiple patient measurements simultaneously, machine learning algorithms can provide an additional source of information that complements traditional medical assessment.

## Preventative Healthcare

Preventing cardiovascular disease is considerably more effective and less expensive than treating advanced cardiac events.

Accurate risk prediction allows healthcare providers to focus preventative resources on patients who are most likely to benefit from early intervention.

## Cost Reduction

Emergency treatment for cardiovascular disease often requires hospitalization, specialized procedures, and long-term rehabilitation.

Earlier detection of high-risk patients can reduce healthcare costs by decreasing emergency admissions and supporting preventative care strategies.

## Scalability

Machine learning systems can evaluate thousands of patient records quickly and consistently, making them valuable tools for hospitals, clinics, and healthcare organizations managing large patient populations.

---

# Dataset

## Dataset Source

**Heart Disease Classification Dataset**

**Kaggle:** https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset/data

The dataset contains structured clinical information collected from **1,319 patients** and is intended for binary heart disease classification.

Each observation represents one patient and contains measurements commonly used during cardiovascular assessment.

The dataset consists of **eight predictive features** describing patient demographics, vital signs, and laboratory biomarkers, along with one target variable indicating whether the patient experienced a heart attack.

Because the target variable contains only two possible outcomes (Positive and Negative), the problem is formulated as a **binary supervised classification task**.

## Dataset Statistics

| Property | Value |
|-----------|------:|
| Total Records | 1,319 |
| Predictive Features | 8 |
| Target Variable | 1 |
| Positive Cases | 810 |
| Negative Cases | 509 |
| Classification Type | Binary |

## Feature Description

| Feature | Data Type | Description | Clinical Significance |
|----------|-----------|-------------|-----------------------|
| Age | Continuous | Patient age | Cardiovascular risk generally increases with age. |
| Gender | Categorical | Female = 0, Male = 1 | Male patients often experience cardiovascular disease at younger ages. |
| Impulse | Continuous | Heart rate | Abnormal heart rate may indicate cardiovascular stress. |
| Pressure High | Continuous | Systolic blood pressure | Elevated systolic pressure is a major cardiovascular risk factor. |
| Pressure Low | Continuous | Diastolic blood pressure | Abnormally low diastolic pressure may indicate cardiovascular complications. |
| Glucose | Continuous | Blood glucose concentration | Elevated glucose is strongly associated with diabetes and cardiovascular disease. |
| CK-MB | Continuous | Creatine Kinase-MB | Cardiac enzyme released following heart muscle damage. |
| Troponin | Continuous | Cardiac troponin concentration | One of the primary biomarkers used clinically to diagnose heart attacks. |
| Class | Target | Positive / Negative | Indicates the presence or absence of heart attack risk. |

---
# Dataset Exploration

Before building predictive models, the dataset was carefully explored to understand its structure, verify data quality, identify potential preprocessing requirements, and evaluate class distribution.

Understanding the dataset before model development is an essential step in every machine learning workflow because issues such as missing values, duplicate records, incorrect data types, or severe class imbalance can significantly affect model performance.

## Binary Classification Task

The target column, **Class**, contains only two unique values:

| Target Label | Meaning |
|--------------|---------|
| Positive | Patient is classified as having heart attack risk |
| Negative | Patient is classified as not having heart attack risk |

Since the objective is to predict one of two possible categories rather than estimate a continuous numerical value, this problem is classified as a **binary supervised classification task**.

---

## Dataset Inspection

The dataset was inspected before any preprocessing operations were performed.

### Dataset Dimensions

| Property | Value |
|-----------|------:|
| Rows | 1,319 |
| Columns | 9 |

The dataset includes one target variable and eight predictive features.

---

### Class Distribution

Understanding the distribution of the target classes is particularly important in medical machine learning because many healthcare datasets are naturally imbalanced.

| Class | Number of Samples |
|--------|------------------:|
| Positive | 810 |
| Negative | 509 |

Although both classes are reasonably represented, the dataset is **moderately imbalanced**, with considerably more positive cases than negative cases.

The imbalance ratio is approximately:

```
Minority Class / Majority Class

509 / 810 ≈ 0.63
```

An imbalance ratio of **0.63** indicates that the minority class contains approximately 63% as many observations as the majority class.

While this imbalance is not extreme, it can still influence the learning behavior of many classification algorithms.

Models trained directly on imbalanced datasets may become biased toward predicting the majority class because doing so minimizes the overall training error.

In healthcare applications, this behavior is particularly undesirable because different prediction errors have very different consequences.

- **False Positives** may lead to unnecessary medical examinations or additional testing.
- **False Negatives** may fail to identify patients who genuinely require medical attention.

Since false negatives can have severe clinical consequences, handling class imbalance was considered an important part of this study.

---

## Data Quality Assessment

High-quality input data is fundamental to producing reliable machine learning models.

The dataset was therefore examined for several common data quality issues before model development.

### Missing Values

Missing values often require imputation or removal before training.

After inspection, **no missing values were identified**.

| Inspection | Result |
|------------|--------|
| Missing Values | None |

Because the dataset was complete, no imputation techniques were necessary.

---

### Duplicate Records

Duplicate observations can bias machine learning algorithms by giving certain samples disproportionate influence during training.

The dataset was inspected for duplicate patient records.

| Inspection | Result |
|------------|--------|
| Duplicate Records | None |

No duplicate observations were detected.

---

### Invalid Data

All variables contained valid numerical or categorical values within expected ranges.

No invalid entries required correction.

---

### Data Leakage

Potential sources of data leakage were also considered before training.

No predictor variable directly revealed the target class.

Sampling techniques were later applied **only to the training dataset**, ensuring that the testing data remained completely independent throughout model evaluation.

---

# Methodology

The overall machine learning workflow followed throughout this project is summarized below.

```
Dataset
    │
    ▼
Data Exploration
    │
    ▼
Data Preprocessing
    │
    ▼
Train-Test Split
    │
    ▼
Sampling Strategies
    │
    ▼
Model Training
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Performance Evaluation
    │
    ▼
Model Comparison
```

Each stage was designed to ensure fair comparison between algorithms while minimizing bias and preventing data leakage.

---

# Data Preprocessing

Data preprocessing transforms raw clinical data into a format suitable for machine learning algorithms.

Although the dataset was already relatively clean, several preprocessing steps were necessary to improve model performance and ensure consistent numerical representation across all features.

---

## Target Encoding

Most machine learning algorithms require numerical target labels.

The original target column contained categorical values.

| Original Label | Encoded Value |
|---------------|--------------:|
| Positive | 1 |
| Negative | 0 |

Encoding the target variable into binary numerical values allows classification algorithms to optimize mathematical decision boundaries during training.

---

## Feature Scaling

The dataset contains measurements collected on very different numerical scales.

For example:

| Feature | Typical Range |
|----------|--------------:|
| Age | 20–90 |
| Blood Pressure | 60–250 |
| Glucose | 50–400 |
| Troponin | Decimal values |

Without normalization, variables with larger numerical ranges can dominate the optimization process.

This is particularly important for gradient-based algorithms such as:

- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM)

To eliminate this issue, **Min-Max Normalization** was applied.

The transformation is defined as:

```
x_normalized = (x − x_min) / (x_max − x_min)
```

This rescales every continuous feature into the interval:

```
[0,1]
```

while preserving the relationships between observations.

The categorical variables (`Gender` and `Class`) were intentionally excluded from normalization because their numerical values already represent discrete categories.

---

## Why Normalize?

Normalization provides several important benefits.

- Prevents variables with large numerical ranges from dominating training.
- Accelerates gradient-based optimization.
- Improves convergence during neural network training.
- Produces more stable Support Vector Machine decision boundaries.
- Ensures features contribute more equally during model learning.

Although tree-based algorithms such as Decision Trees and Random Forests do not require feature scaling, the same preprocessing pipeline was maintained across all experiments for consistency.

---

# Handling Class Imbalance

Medical datasets frequently exhibit class imbalance because disease prevalence is rarely distributed equally across populations.

Although the imbalance in this dataset is moderate, several sampling strategies were investigated to determine whether balancing the classes could improve predictive performance.

Three different training datasets were evaluated.

| Dataset | Sampling Strategy |
|----------|-------------------|
| Original Dataset | None |
| SMOTE | Oversampling |
| NearMiss | Undersampling |

Each machine learning model was independently trained using all three datasets.

This allowed the influence of sampling strategy to be evaluated separately from the learning algorithm itself.

---

# Sampling Techniques

## Original Dataset

The first training approach used the dataset exactly as collected.

No synthetic observations were added, and no existing records were removed.

Advantages include:

- Preserves all original information.
- Maintains the true clinical distribution.
- Avoids introducing artificial observations.
- Prevents information loss.

This dataset served as the baseline for comparison against all resampling methods.

---

## SMOTE (Synthetic Minority Oversampling Technique)

SMOTE is one of the most widely used oversampling algorithms for imbalanced classification problems.

Instead of duplicating minority-class observations, SMOTE generates entirely new synthetic examples by interpolating between neighboring minority samples.

This increases the representation of the minority class while preserving diversity within the data.

### Advantages

- Balances the dataset without simple duplication.
- Improves minority-class representation.
- Often increases recall.
- Preserves majority-class observations.

### Potential Limitations

- Synthetic observations may introduce noise.
- Can increase false-positive predictions.
- May reduce precision if synthetic samples overlap existing classes.

---

## NearMiss

NearMiss is an undersampling technique designed to balance datasets by selectively removing observations from the majority class.

Rather than deleting samples randomly, NearMiss retains majority-class observations that lie closest to minority-class examples.

The resulting dataset contains equal numbers of observations from both classes.

### Advantages

- Produces balanced datasets.
- Removes redundant majority samples.
- Can improve class separation.

### Potential Limitations

- Removes real patient data.
- May discard valuable information.
- Can reduce overall predictive performance.
- Higher variance because of smaller training datasets.

---

## Sampling Strategy Comparison

| Method | Category | Preserves Original Data | Creates Synthetic Data |
|---------|----------|-------------------------|------------------------|
| Original Dataset | None | ✓ | ✗ |
| SMOTE | Oversampling | ✓ | ✓ |
| NearMiss | Undersampling | ✗ | ✗ |

The effectiveness of each sampling strategy was evaluated independently for every machine learning algorithm.

---

# Train-Test Split

To evaluate model generalization fairly, the dataset was divided into separate training and testing subsets.

| Dataset | Percentage |
|----------|-----------:|
| Training Set | 80% |
| Testing Set | 20% |

A **stratified train-test split** was used.

Stratification preserves approximately the same class distribution in both datasets, ensuring that the testing data accurately represents the original population.

To guarantee unbiased evaluation, sampling techniques were applied **only after the training dataset had been separated**.

The testing dataset remained completely untouched throughout preprocessing and model development.

This prevents **data leakage**, ensuring that evaluation metrics accurately reflect how each model performs on previously unseen patient data.

---

# Machine Learning Models

Five supervised learning algorithms were selected for comparison.

Each model represents a different family of machine learning techniques and offers unique advantages for structured clinical datasets.

| Model | Learning Category |
|--------|-------------------|
| Multi-Layer Perceptron (MLP) | Artificial Neural Network |
| Support Vector Machine (SVM) | Margin-Based Classifier |
| Decision Tree | Tree-Based Learning |
| Random Forest | Bagging Ensemble |
| AdaBoost | Boosting Ensemble |

Rather than relying on a single algorithm, comparing multiple learning approaches provides a more comprehensive understanding of which techniques are most suitable for heart attack risk prediction.

Each model was independently optimized using hyperparameter tuning and evaluated under identical experimental conditions.
# Evaluation Metrics

Evaluating machine learning models solely by accuracy can be misleading, especially in healthcare applications where the consequences of different prediction errors vary significantly.

For example, a model that predicts every patient as positive may achieve high recall but would also generate an unacceptable number of false alarms. Conversely, a model with extremely high precision may fail to identify patients who actually require medical attention.

To obtain a comprehensive understanding of model performance, multiple evaluation metrics were used throughout this project.

| Metric | Formula | Purpose |
|---------|---------|---------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Measures the overall percentage of correctly classified samples. |
| Precision | TP / (TP + FP) | Measures how many predicted positive cases are actually positive. |
| Recall (Sensitivity) | TP / (TP + FN) | Measures how many actual positive cases are correctly identified. |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | Balances precision and recall into a single metric. |
| Balanced Accuracy | (Sensitivity + Specificity) / 2 | Accounts for class imbalance by averaging recall across both classes. |
| ROC Curve | TPR vs FPR | Evaluates discrimination ability across different thresholds. |
| ROC-AUC | Area under the ROC Curve | Measures the model's ability to distinguish between classes. |
| Precision-Recall Curve | Precision vs Recall | More informative for imbalanced classification problems. |
| PR-AUC | Area under the Precision-Recall Curve | Measures overall precision-recall performance. |
| Confusion Matrix | [[TN, FP], [FN, TP]] | Summarizes all correct and incorrect predictions. |

## Why Multiple Metrics Matter

Healthcare classification problems require balancing several competing objectives.

- **High Precision** reduces unnecessary medical examinations by minimizing false positives.
- **High Recall** ensures that patients at genuine risk are less likely to be overlooked.
- **High F1-Score** indicates a good compromise between precision and recall.
- **Balanced Accuracy** provides a fair evaluation when class distributions are uneven.
- **ROC-AUC** measures overall class discrimination regardless of classification threshold.
- **PR-AUC** focuses specifically on positive-class performance and is often more informative for medical datasets with class imbalance.

Using multiple evaluation metrics provides a much more reliable assessment than relying on overall accuracy alone.

---

# Hyperparameter Tuning

Each machine learning algorithm contains parameters that influence how the model learns from data.

Rather than using default settings, multiple hyperparameter combinations were evaluated to determine the configuration that produced the strongest predictive performance.

Every model was tuned independently while maintaining the same train-test split and evaluation procedure to ensure fair comparison.

---

# Model Results

The following sections summarize the performance of each machine learning model using the three sampling strategies evaluated in this project.

For every classifier, the following information is presented:

- Hyperparameters tested
- Best hyperparameter configuration
- Performance metrics
- Model discussion
- Interpretation of results

---

# Multi-Layer Perceptron (MLP)

## Overview

The Multi-Layer Perceptron (MLP) is a feedforward artificial neural network designed to learn complex nonlinear relationships between input variables.

Unlike linear classifiers, neural networks automatically learn hierarchical feature representations through multiple hidden layers, allowing them to capture interactions between clinical variables that may not be immediately obvious.

The model was trained using Binary Crossentropy loss and optimized using the Adam optimizer. Early stopping was incorporated during training to reduce overfitting by monitoring validation performance.

---

## Hyperparameter Optimization

| Hyperparameter | Values Tested | Best Value |
|----------------|---------------|------------|
| Hidden Layers | 1, 2, 3 | 2 |
| Neurons per Layer | 16, 32, 64 | 32 |
| Activation Function | ReLU, Tanh, Sigmoid | ReLU |
| Optimizer | SGD, Adam | Adam |
| Learning Rate | 0.001, 0.01, 0.1 | 0.001 |
| Loss Function | Binary Crossentropy | Binary Crossentropy |
| Batch Size | 16, 32, 64 | 32 |
| Epochs | 50, 100, 150 | 100 |
| Early Stopping | Yes / No | Yes |

---

## Performance Comparison

| Sampling Strategy | Test Loss | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|------------------|----------:|---------:|----------:|--------:|----------:|------------------:|--------:|-------:|
| Original Dataset | **0.468** | **0.807** | **0.840** | **0.846** | **0.843** | **0.795** | **0.868** | **0.921** |
| SMOTE | 0.679 | 0.614 | 0.614 | **1.000** | 0.761 | 0.500 | 0.510 | 0.594 |
| NearMiss | 0.715 | 0.621 | 0.641 | 0.870 | 0.738 | 0.548 | 0.505 | 0.597 |

---

## Discussion

Among all three sampling strategies, training on the **original dataset** produced the strongest overall performance.

The model achieved an accuracy of **80.7%**, together with balanced precision, recall, and F1-score, indicating that it generalized well despite the moderate class imbalance.

Applying SMOTE substantially increased recall, allowing the network to correctly identify nearly every positive case. However, this improvement came at the expense of precision, balanced accuracy, ROC-AUC, and PR-AUC. The model became overly sensitive to the positive class, producing many false-positive predictions.

NearMiss performed slightly better than SMOTE but still reduced overall predictive performance. Because undersampling removes genuine patient records, valuable information available in the original dataset was lost, limiting the neural network's ability to learn robust decision boundaries.

Overall, the original dataset allowed the MLP to achieve the most reliable balance between sensitivity and specificity.

---

## Interpretation

The confusion matrix demonstrated that most patient records were classified correctly, with relatively few false positives and false negatives.

The training and validation loss curves decreased together throughout training, indicating stable optimization and minimal evidence of overfitting. Early stopping successfully prevented unnecessary training once validation performance stabilized.

The ROC curve (AUC = **0.868**) and Precision-Recall curve (AUC = **0.921**) further demonstrated that the model maintained strong discrimination between positive and negative cases despite the moderately imbalanced dataset.

> **Figure Placeholder**
>
> - Training and Validation Loss
> - Confusion Matrix
> - ROC Curve
> - Precision-Recall Curve

---

# Support Vector Machine (SVM)

## Overview

Support Vector Machines classify observations by identifying the optimal decision boundary that maximizes the margin separating different classes.

Instead of attempting to classify every training sample perfectly, SVMs focus on finding the boundary that provides the greatest separation between positive and negative observations.

Because clinical data often exhibit nonlinear relationships, an **RBF (Radial Basis Function)** kernel was selected after hyperparameter tuning.

---

## Hyperparameter Optimization

| Hyperparameter | Values Tested | Best Value |
|----------------|---------------|------------|
| Kernel | Linear, Polynomial, RBF, Sigmoid | RBF |
| Regularization (C) | 0.1, 1.0, 10.0 | 1.0 |
| Gamma | Scale, Auto, 0.01, 0.1 | Scale |
| Class Weight | None, Balanced | None |
| Probability | True / False | False |
| Random State | 0, 42, 123 | 42 |

---

## Performance Comparison

| Sampling Strategy | Test Loss | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|------------------|----------:|---------:|----------:|--------:|----------:|------------------:|--------:|-------:|
| Original Dataset | **0.506** | **0.742** | 0.787 | **0.796** | **0.791** | 0.727 | 0.829 | 0.893 |
| SMOTE | 0.510 | 0.727 | **0.869** | 0.654 | 0.746 | **0.749** | **0.853** | **0.907** |
| NearMiss | 0.582 | 0.686 | 0.793 | 0.660 | 0.721 | 0.693 | 0.755 | 0.838 |

---

## Discussion

The Support Vector Machine demonstrated consistent performance across all sampling strategies but proved more sensitive to class balancing than the neural network.

Training on the original dataset produced the highest overall accuracy, recall, and F1-score, indicating that the natural class distribution enabled the classifier to construct a more representative decision boundary.

SMOTE produced the highest precision, ROC-AUC, and PR-AUC, suggesting improved separation between classes. However, this came with reduced recall, indicating that the classifier became more conservative when identifying positive patients.

NearMiss yielded the weakest overall performance. Removing majority-class observations reduced the amount of information available during training, leading to lower overall accuracy and F1-score.

Overall, the original dataset again produced the strongest balance between predictive performance and generalization.

---

## Interpretation

The confusion matrix showed that most patient records were classified correctly, with relatively few false-positive and false-negative predictions.

The ROC-AUC of **0.829** demonstrated good class discrimination, while the PR-AUC of **0.893** indicated strong precision-recall performance for the positive class.

Visualization of the decision boundary using **Age** and **Troponin** illustrated how the RBF kernel captured nonlinear relationships that would not have been possible using a linear classifier.

> **Figure Placeholder**
>
> - Confusion Matrix
> - ROC Curve
> - Precision-Recall Curve
> - SVM Decision Boundary
# Decision Tree

## Overview

Decision Trees are supervised learning algorithms that recursively partition data into increasingly homogeneous groups by selecting feature thresholds that maximize class separation.

Unlike neural networks or Support Vector Machines, Decision Trees are highly interpretable because every prediction can be traced through a sequence of simple decision rules. This transparency makes them particularly attractive for healthcare applications where understanding how predictions are made is often just as important as prediction accuracy.

However, individual Decision Trees are prone to overfitting because they can continue splitting until they memorize the training data. Hyperparameter tuning was therefore performed to balance model complexity and generalization.

---

## Hyperparameter Optimization

| Hyperparameter | Values Tested | Best Value |
|----------------|---------------|------------|
| Criterion | Gini, Entropy | Gini |
| Maximum Depth | None, 3, 5, 10 | None |
| Minimum Samples Split | 2, 5, 10 | 2 |
| Minimum Samples Leaf | 1, 2, 5 | 1 |
| Class Weight | None, Balanced | None |
| Random State | 0, 42, 123 | 42 |

---

## Performance Comparison

| Sampling Strategy | Test Loss | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|------------------|----------:|---------:|----------:|--------:|----------:|------------------:|--------:|-------:|
| Original Dataset | 0.683 | 0.981 | 0.982 | 0.988 | 0.985 | 0.979 | 0.979 | 0.977 |
| SMOTE | **0.672** | **0.982** | **0.983** | **0.988** | **0.986** | **0.980** | **0.980** | **0.978** |
| NearMiss | 0.819 | 0.977 | 0.981 | 0.981 | 0.981 | 0.976 | 0.976 | 0.975 |

---

## Discussion

The Decision Tree achieved exceptionally strong classification performance across all sampling strategies, with accuracy consistently exceeding **97%**.

Training on the original dataset and the SMOTE-balanced dataset produced nearly identical results, indicating that the classifier was relatively insensitive to moderate class imbalance. Among the three approaches, SMOTE produced marginal improvements in overall accuracy and balanced accuracy while also yielding the lowest test loss.

NearMiss resulted in slightly lower performance across nearly every metric. Because NearMiss removes majority-class observations, some clinically meaningful information was discarded during training, reducing the tree's ability to identify optimal decision boundaries.

Although Decision Trees achieved outstanding predictive accuracy, they remain susceptible to overfitting because a single tree can become highly specialized to the training data.

---

## Interpretation

The confusion matrix demonstrated excellent classification performance, with only a handful of false-positive and false-negative predictions.

The ROC-AUC and PR-AUC values, both approximately **0.98**, indicate outstanding class discrimination and confirm that the model separates positive and negative patients effectively.

One of the primary strengths of Decision Trees is interpretability. Each prediction follows a sequence of explicit feature thresholds, making it possible to understand how individual patient characteristics influence classification outcomes.

> **Figure Placeholder**
>
> - Decision Tree Visualization
> - Confusion Matrix
> - ROC Curve
> - Precision-Recall Curve

---

# Random Forest

## Overview

Random Forest is an ensemble learning algorithm that combines the predictions of many Decision Trees to produce a single, more stable classifier.

Instead of relying on one tree, Random Forest trains multiple trees using bootstrap sampling and randomly selected feature subsets. Final predictions are determined through majority voting across the ensemble.

This approach substantially reduces the high variance associated with individual Decision Trees while improving predictive accuracy and generalization.

Because Random Forest naturally handles nonlinear relationships, feature interactions, and moderate class imbalance, it is widely regarded as one of the strongest algorithms for structured tabular datasets.

---

## Hyperparameter Optimization

| Hyperparameter | Values Tested | Best Value |
|----------------|---------------|------------|
| Number of Trees | 50, 100, 200 | 100 |
| Maximum Depth | None, 5, 10, 20 | None |
| Minimum Samples Split | 2, 5, 10 | 2 |
| Minimum Samples Leaf | 1, 2, 4 | 1 |
| Maximum Features | Auto, sqrt, log2 | Auto |
| Bootstrap | True, False | True |
| Random State | 0, 42, 123 | 42 |

---

## Performance Comparison

| Sampling Strategy | Test Loss | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|------------------|----------:|---------:|----------:|--------:|----------:|------------------:|--------:|-------:|
| Original Dataset | 0.093 | **0.985** | **0.988** | **0.988** | **0.988** | **0.984** | **0.994** | **0.996** |
| SMOTE | **0.087** | **0.985** | **0.988** | **0.988** | **0.988** | **0.984** | 0.993 | 0.994 |
| NearMiss | 0.113 | 0.981 | 0.988 | 0.981 | 0.985 | 0.981 | 0.990 | 0.994 |

---

## Discussion

Random Forest consistently produced the strongest overall performance of every model evaluated in this project.

Classification accuracy reached **98.5%**, while precision, recall, and F1-score all approached **0.99**, demonstrating excellent predictive performance.

Although SMOTE produced the lowest test loss, the original dataset achieved slightly higher ROC-AUC and PR-AUC values, indicating marginally better class discrimination.

Interestingly, the differences between the original dataset and SMOTE were extremely small, suggesting that Random Forest naturally handles moderate class imbalance without requiring aggressive resampling techniques.

NearMiss also performed well but showed slight reductions across several evaluation metrics due to information loss caused by undersampling.

Overall, Random Forest demonstrated outstanding robustness regardless of the sampling strategy employed.

---

## Interpretation

Random Forest significantly reduced the overfitting tendency observed in a single Decision Tree by averaging predictions across many independently trained trees.

The confusion matrix contained only a few misclassified observations, confirming the model's exceptional predictive capability.

ROC-AUC values approaching **1.0** indicate near-perfect separation between positive and negative classes.

These results demonstrate why ensemble learning methods frequently outperform individual classifiers on structured healthcare datasets.

> **Figure Placeholder**
>
> - Confusion Matrix
> - ROC Curve
> - Precision-Recall Curve
> - Random Forest Feature Importance

---

# AdaBoost

## Overview

AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak learners into a stronger classifier.

Rather than training every model independently, AdaBoost trains decision trees sequentially. After each iteration, the algorithm increases the weight assigned to previously misclassified observations, encouraging subsequent learners to focus on the most difficult cases.

This iterative refinement often leads to excellent predictive accuracy while maintaining relatively simple individual models.

---

## Hyperparameter Optimization

| Hyperparameter | Values Tested | Best Value |
|----------------|---------------|------------|
| Number of Estimators | 50, 100, 200 | 50 |
| Base Estimator Depth | 1, 2, 3 | 1 |
| Learning Rate | 0.01, 0.1, 1.0 | 1.0 |
| Random State | 0, 42, 123 | 42 |

---

## Performance Comparison

| Sampling Strategy | Test Loss | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|------------------|----------:|---------:|----------:|--------:|----------:|------------------:|--------:|-------:|
| Original Dataset | 0.427 | 0.985 | 0.988 | 0.988 | 0.988 | 0.984 | 0.985 | 0.989 |
| SMOTE | 0.430 | 0.985 | 0.988 | 0.988 | 0.988 | 0.984 | 0.985 | 0.989 |
| NearMiss | **0.425** | **0.985** | **0.988** | **0.988** | **0.988** | **0.984** | 0.983 | 0.988 |

---

## Discussion

AdaBoost produced remarkably consistent results across all three sampling strategies.

Classification accuracy remained approximately **98.5%** regardless of whether the original dataset, SMOTE, or NearMiss was used.

NearMiss achieved the lowest test loss, although the differences between all three datasets were extremely small. This indicates that AdaBoost generalized effectively regardless of class balancing strategy.

The algorithm's ability to iteratively focus on difficult observations contributed to its consistently strong predictive performance.

Overall, AdaBoost ranked alongside Random Forest as one of the strongest classifiers evaluated in this study.

---

## Interpretation

The confusion matrix showed only a few incorrectly classified patients, reflecting the model's excellent predictive accuracy.

Both ROC-AUC and PR-AUC remained exceptionally high, confirming strong class discrimination and an excellent balance between precision and recall.

The shallow decision trees used as weak learners prevented excessive model complexity while still allowing AdaBoost to capture meaningful patterns within the clinical data.

> **Figure Placeholder**
>
> - Confusion Matrix
> - ROC Curve
> - Precision-Recall Curve
> - AdaBoost Performance Visualization

---
# Feature Importance

Understanding **why** a machine learning model makes a particular prediction is just as important as achieving high predictive accuracy, especially in healthcare applications where clinical interpretability is essential.

Tree-based models such as **Decision Trees** and **Random Forests** naturally provide feature importance scores that estimate how much each feature contributes to the prediction process.

Feature importance is calculated by measuring how much each variable reduces impurity when used to split the data throughout the decision trees. Features that consistently create better splits receive higher importance scores.

Analyzing these scores helps determine which clinical measurements have the greatest influence on heart attack prediction.

---

## Most Influential Features

The models consistently identified the following variables as the strongest predictors of heart attack risk.

| Rank | Feature | Clinical Importance |
|-----:|---------|---------------------|
| 1 | Troponin | Primary biomarker used to diagnose heart muscle damage. |
| 2 | CK-MB | Cardiac enzyme released during myocardial injury. |
| 3 | Glucose | Elevated glucose is associated with diabetes and increased cardiovascular risk. |
| 4 | Systolic Blood Pressure | High blood pressure significantly increases cardiovascular risk. |
| 5 | Age | Cardiovascular disease becomes more prevalent with increasing age. |
| 6 | Heart Rate (Impulse) | Abnormal heart rate may indicate cardiovascular stress. |
| 7 | Diastolic Blood Pressure | Provides additional cardiovascular information when combined with systolic pressure. |
| 8 | Gender | Influences long-term cardiovascular risk but contributed less than physiological measurements. |

---

## Discussion

The feature importance results align closely with established medical knowledge.

**Troponin** emerged as the most influential predictor across the tree-based models. This finding is expected because cardiac troponin is one of the primary laboratory biomarkers used by physicians to diagnose myocardial infarction and assess heart muscle damage.

**CK-MB (Creatine Kinase-MB)** ranked second in importance. Similar to troponin, CK-MB is released into the bloodstream following injury to cardiac muscle tissue, making it another clinically significant indicator of heart attacks.

Other physiological variables—including blood glucose, systolic blood pressure, heart rate, and age—also contributed substantially to model predictions. These measurements are widely recognized as important cardiovascular risk factors and collectively provide valuable context when estimating a patient's likelihood of experiencing a cardiac event.

Overall, the feature importance analysis demonstrates that the machine learning models relied on medically meaningful variables rather than arbitrary statistical relationships. This agreement between model behavior and established clinical knowledge increases confidence in the predictive results.

> **Figure Placeholder**
>
> - Random Forest Feature Importance
> - Decision Tree Feature Importance

---

# Overall Model Comparison

The following table summarizes the best-performing configuration for each machine learning algorithm evaluated throughout this project.

| Model | Best Sampling Strategy | Test Loss | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|--------|-----------------------|----------:|---------:|----------:|--------:|----------:|------------------:|--------:|-------:|
| Multi-Layer Perceptron | Original Dataset | 0.468 | 0.807 | 0.840 | 0.846 | 0.843 | 0.795 | 0.868 | 0.921 |
| Support Vector Machine | Original Dataset | 0.506 | 0.742 | 0.787 | 0.796 | 0.791 | 0.727 | 0.829 | 0.893 |
| Decision Tree | SMOTE | 0.672 | 0.982 | 0.983 | 0.988 | 0.986 | 0.980 | 0.980 | 0.978 |
| Random Forest | SMOTE* | **0.087** | **0.985** | **0.988** | **0.988** | **0.988** | **0.984** | **0.994** | **0.996** |
| AdaBoost | NearMiss | 0.425 | 0.985 | 0.988 | 0.988 | 0.988 | 0.984 | 0.983 | 0.988 |

> **Note:** Although SMOTE produced the lowest test loss for Random Forest, the original dataset achieved slightly higher ROC-AUC and PR-AUC scores. Both configurations performed exceptionally well, indicating that Random Forest is highly robust to moderate class imbalance.

---

## Performance Summary

Several important observations emerged during model comparison.

### Ensemble Learning Outperformed Individual Models

The strongest predictive performance came from the ensemble methods.

Both **Random Forest** and **AdaBoost** consistently achieved approximately **98.5% accuracy**, excellent F1-scores, and outstanding ROC-AUC values.

These algorithms benefit from combining multiple weak learners into a single predictive model, substantially reducing variance while improving generalization.

---

### Random Forest Produced the Strongest Overall Performance

Random Forest achieved the highest overall performance across nearly every evaluation metric.

Its advantages include:

- Highest overall accuracy
- Lowest test loss
- Excellent ROC-AUC
- Excellent PR-AUC
- Strong robustness to sampling strategy
- Minimal evidence of overfitting

The combination of bootstrap aggregation (bagging) and random feature selection allowed the model to learn highly accurate decision boundaries while avoiding excessive model complexity.

---

### AdaBoost Performed Nearly as Well

AdaBoost produced performance almost identical to Random Forest.

Its sequential boosting strategy enabled the classifier to focus progressively on difficult training examples while maintaining excellent generalization.

The extremely small differences between AdaBoost and Random Forest suggest that both ensemble learning approaches are highly suitable for structured clinical datasets.

---

### Decision Tree Achieved Excellent Accuracy

Despite being a single-tree model, the Decision Tree achieved outstanding classification performance.

However, unlike ensemble methods, individual Decision Trees are naturally more susceptible to overfitting because they rely on a single hierarchical structure.

Although pruning and hyperparameter tuning help reduce this issue, ensemble methods generally provide more stable predictions.

---

### Neural Network Performance

The Multi-Layer Perceptron produced respectable performance with an accuracy of approximately **81%**.

Although considerably lower than the tree-based models, the neural network still demonstrated good predictive capability.

One possible explanation is that relatively small structured datasets often favor tree-based algorithms over deep learning approaches, which typically require substantially larger datasets to fully leverage their representational capacity.

---

### Support Vector Machine Performance

The Support Vector Machine produced the lowest overall accuracy among the evaluated models.

Nevertheless, the classifier still demonstrated good precision, recall, and discrimination ability.

The nonlinear RBF kernel successfully captured complex feature relationships, although its performance remained below that of the ensemble methods.

---

# Overfitting Analysis

One of the primary goals of model evaluation is determining whether a classifier generalizes well beyond the training data.

High training accuracy alone does not guarantee that a model will perform well on unseen patients.

Each algorithm was therefore evaluated for signs of overfitting.

---

## Multi-Layer Perceptron

The neural network exhibited minimal evidence of overfitting.

Training and validation loss decreased smoothly throughout optimization, with both curves converging toward similar values.

Early stopping prevented unnecessary training after validation performance stabilized, reducing the likelihood of memorizing the training data.

---

## Support Vector Machine

The Support Vector Machine generalized well across the testing dataset.

The relatively small difference between training and testing performance suggests that the selected regularization parameter successfully controlled model complexity.

The margin-maximization principle inherent to SVMs naturally contributes to strong generalization.

---

## Decision Tree

Among all evaluated algorithms, the Decision Tree displayed the greatest tendency toward overfitting.

Single trees can continue splitting until they memorize training observations, resulting in extremely high training accuracy while sacrificing some generalization performance.

Although hyperparameter tuning helped mitigate this issue, the model remained more sensitive to overfitting than the ensemble approaches.

---

## Random Forest

Random Forest substantially reduced overfitting by averaging predictions from multiple independently trained Decision Trees.

Because each tree observes a different bootstrap sample and random subset of features, individual prediction errors tend to cancel out during majority voting.

This ensemble strategy explains the model's outstanding balance between accuracy and generalization.

---

## AdaBoost

AdaBoost demonstrated strong generalization with only minimal evidence of overfitting.

Using shallow decision trees as weak learners prevented the ensemble from becoming excessively complex while still allowing the algorithm to focus on difficult training examples.

Although boosting methods can become sensitive to noisy observations, the selected hyperparameters maintained an effective balance between model complexity and predictive performance.

---
# Conclusion

This project demonstrated that supervised machine learning can effectively predict heart attack risk using structured clinical data consisting of demographic information, vital signs, and cardiac biomarkers.

Beginning with a dataset of **1,319 patient records**, a complete machine learning pipeline was developed that included data exploration, preprocessing, feature scaling, handling class imbalance, model training, hyperparameter optimization, and comprehensive model evaluation.

Five different classification algorithms were investigated:

- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost

Each model was trained using three different sampling strategies (Original Dataset, SMOTE, and NearMiss) to evaluate the effect of class balancing on predictive performance.

Among all evaluated models, **Random Forest** consistently achieved the strongest overall performance, reaching approximately **98.5% accuracy**, **0.988 precision**, **0.988 recall**, **0.988 F1-score**, and an outstanding **ROC-AUC of 0.994**. AdaBoost produced nearly identical results, further demonstrating the effectiveness of ensemble learning techniques for structured medical datasets.

The experiments also revealed that moderate class imbalance did not significantly degrade the performance of tree-based ensemble models. In many cases, training on the original dataset produced results that were as good as—or even slightly better than—those obtained after resampling. This suggests that robust ensemble methods such as Random Forest naturally handle moderate imbalance without requiring aggressive oversampling or undersampling.

Feature importance analysis further validated the models by identifying **Troponin** and **CK-MB** as the two most influential predictors. These biomarkers are widely recognized in clinical practice as primary indicators of cardiac injury, providing confidence that the models relied on medically meaningful information rather than arbitrary statistical correlations.

Overall, this project demonstrates that carefully designed machine learning models can achieve highly accurate heart attack risk prediction while remaining interpretable, robust, and consistent with established clinical knowledge.

---

# Key Takeaways

- Structured clinical data can be highly effective for heart attack risk prediction.
- Proper preprocessing and feature scaling improve model stability and consistency.
- Moderate class imbalance does not necessarily require aggressive resampling when using robust ensemble methods.
- Random Forest and AdaBoost substantially outperformed the neural network and Support Vector Machine on this dataset.
- Tree-based ensemble models provided both exceptional predictive performance and strong generalization.
- Troponin and CK-MB emerged as the most important predictive features, aligning closely with established medical practice.
- Multiple evaluation metrics should always be considered when assessing healthcare classification models rather than relying solely on accuracy.

---

# References

## Dataset

Bharath_011. **Heart Disease Classification Dataset.** Kaggle.

https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset/data

---

## World Health Organization

World Health Organization.

**Cardiovascular diseases (CVDs)**

https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

---

## Scikit-Learn Documentation

Decision Trees

https://scikit-learn.org/stable/modules/tree.html

Model Evaluation

https://scikit-learn.org/stable/modules/model_evaluation.html

---

## TensorFlow

https://www.tensorflow.org/

---

## Pandas

https://pandas.pydata.org/

---

## NumPy

https://numpy.org/

---

## Matplotlib

https://matplotlib.org/

---

## imbalanced-learn

https://imbalanced-learn.org/

---