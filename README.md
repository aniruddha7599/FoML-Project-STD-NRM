# Classification Models for Predicting Non-Alcoholic Fatty Liver Disease (NAFLD)

---

## Project Overview

This project implements machine learning classification models to predict the presence or absence of Non-Alcoholic Fatty Liver Disease (NAFLD) using simple, widely-available physical parameters (age, gender, BMI, waist circumference, etc.). The goal is to provide an affordable, scalable screening tool that can help healthcare professionals identify individuals at risk and enable early interventions.

The models explored in this work are:

* Gaussian Naive Bayes (GaussianNB)
* Support Vector Machine (SVM) with RBF kernel (hyperparameter-tuned)
* Random Forest (RF)

After preprocessing and extensive EDA, Random Forest achieved the best overall accuracy.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Preprocessing](#preprocessing)
4. [Models & Training](#models--training)
5. [Results](#results)
6. [How to Reproduce](#how-to-reproduce)
7. [Files in this repo (suggested)](#files-in-this-repo-suggested)
8. [Future Work](#future-work)
9. [References](#references)

---

## Dataset

* **Rows:** 605
* **Columns:** 10 (9 features + 1 output)
* **Output (target):** `Fibrosis Status` — 0 = No Fibrosis, 1 = Fibrosis (Fibrosis 1 and above)
* **Output class ratio:** 196 : 409 (0 : 1)

**Features:**

* `Age` (years)
* `Gender` (0 = Female, 1 = Male)
* `Height (cm)`
* `Weight (kg)`
* `Body Mass Index` (BMI)
* `Waist Circumference (cm)`
* `Hip Circumference (cm)`
* `Diabetes` (0 = No, 1 = Yes)
* `Smoking Status` (0 = Not smoking, 1 = Smoking)

**Dataset sources referenced in the report:**

* Raw CSV: `https://raw.githubusercontent.com/aniruddha7599/FoML-Project/refs/heads/main/my_dataframe.csv`
* Kaggle: `https://www.kaggle.com/datasets/tourdeglobe/fatty-liver-disease`

---

## Exploratory Data Analysis (EDA)

Key EDA findings summarized from the report:

* **Missing values:**

  * `Waist Circumference` — 29 missing
  * `Hip Circumference` — 7 missing
  * All other columns: 0 missing

| Column                   | Missing Values |
| ------------------------ | -------------: |
| Age                      |              0 |
| Height (cm)              |              0 |
| Weight (kg)              |              0 |
| Body Mass Index          |              0 |
| Waist Circumference (cm) |             29 |
| Hip Circumference (cm)   |              7 |
| Diabetes                 |              0 |
| Fibrosis Status          |              0 |
| Gender                   |              0 |
| Smoking Status           |              0 |

* **Outlier detection:** IQR-based detection was used to flag outliers. Outlier counts per column (report):

| Column                                        | Outliers Count |
| --------------------------------------------- | -------------: |
| Weight (kg)                                   |             15 |
| Body Mass Index                               |             16 |
| Waist Circumference (cm)                      |             19 |
| Hip Circumference (cm)                        |             11 |
| Age, Gender, Height, Diabetes, Smoking Status |              0 |

* **Observations:** The distribution did not materially change after outlier treatment because the number of outliers was small relative to the dataset size.

* **Linear separability check:** PCA was used to project data and inspect separability. The projection (95% variance capture) suggested the classes are not linearly separable — motivating the use of a non-linear SVM (RBF kernel).

---

## Preprocessing

Steps applied before modeling (as described in the report):

1. **Missing value imputation:** Imputed missing values (waist and hip circumference) with the column mean (due to low missingness).
2. **Outlier detection & treatment:** Detected using IQR; for treatment the 3-sigma rule was applied to label/treat extreme values (report notes that treatment did not materially change distributions).
3. **Standardization:** Features were standardized (zero mean, unit variance) prior to modeling.
4. **Handling class imbalance:** For SVM training, SMOTE (Synthetic Minority Over-sampling Technique) was used to improve learning on the minority class.

---

## Models & Training

### 1. Gaussian Naive Bayes (GaussianNB)

* Assumes feature independence and Gaussian distribution for continuous features.

### 2. Support Vector Machine (SVM)

* Kernel: **RBF** (chosen after PCA showed non-linear separability).
* Hyperparameter tuning using `GridSearchCV` with 5-fold cross-validation.
* Hyperparameters searched (examples used in report):

  * `C`: [0.1, 1, 10, 100]
  * `gamma`: ['auto', 'scale', 0.1, 1]
* Best parameters found (report): `C = 10`, `gamma = 0.1`.
* SMOTE used to address class imbalance during SVM training.

### 3. Random Forest (RF)

* Ensemble of decision trees; reduces overfitting relative to single trees.
* Number of trees used: `n_estimators = 100`.

---

## Results

Evaluation metrics reported for each model (accuracy, precision, recall, F1-score):

| Model                                     |   Accuracy | Precision | Recall | F1-score |
| ----------------------------------------- | ---------: | --------: | -----: | -------: |
| Gaussian Naive Bayes                      |     0.6923 |    0.7742 | 0.7742 |   0.7742 |
| Support Vector Machine (RBF, C=10, γ=0.1) |     0.7363 |    0.7436 | 0.9355 |   0.8286 |
| Random Forest (n=100)                     | **0.7418** |    0.7619 | 0.9032 |   0.8266 |

**Key takeaway:** Random Forest achieved the highest accuracy (74.18%) and competitive precision/recall, outperforming SVM and Naive Bayes on overall accuracy. SVM had the highest recall in this comparison.

---

## How to Reproduce

### Required packages (as used in the project)

```
numpy==1.26.0
pandas==2.1.3
matplotlib==3.8.0
scipy==1.11.3
scikit-learn==1.3.2
imbalanced-learn==0.12.4
```

Install with:

```bash
pip install -r requirements.txt
```

### Suggested steps

1. Clone this repository.
2. Download the dataset (raw CSV) or point the notebook/script to the provided CSV link.
3. Run the preprocessing notebook/script to: impute missing values, standardize features, apply outlier handling, and (optionally) SMOTE.
4. Train the models (GaussianNB, SVM with GridSearchCV, Random Forest) and evaluate using a hold-out test set or cross-validation.

### Example (high-level) commands

```bash
# Start a Jupyter notebook
jupyter notebook

# Or run a training script
python train_models.py --data-path data/my_dataframe.csv
```

**Notes:**

* Use the `GridSearchCV` settings described above for SVM hyperparameter tuning.
* Use `RandomForestClassifier(n_estimators=100)` for the RF baseline.

---

## Conclusion & Future Work

* The Random Forest classifier performed best in this analysis with ~74.18% accuracy.
* Using basic physical parameters (BMI, waist circumference, age, etc.) shows promise for affordable screening of NAFLD in resource-limited settings.

**Future directions:**

* Expand dataset size and diversity.
* Explore additional features (blood tests, liver enzymes, socio-economic factors).
* Try other imbalance-handling strategies and ensemble methods.
* Deploy a lightweight web/mobile screening tool for field usage.

---

## References

1. "40% in India suffer from non-alcoholic fatty liver: Doctors." *Economic Times*.
2. Weidong Ji, Mingyue Xue, Yushan Zhang, Hua Yao, Yushan Wang. "A Machine Learning Based Framework to Identify and Classify Non-alcoholic Fatty Liver Disease in a Large Scale Population." (2022).
3. Fatty Liver Disease dataset on Kaggle: `https://www.kaggle.com/datasets/tourdeglobe/fatty-liver-disease`
