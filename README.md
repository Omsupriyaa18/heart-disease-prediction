# Heart Disease Prediction using Machine Learning

A beginner-friendly machine learning project that predicts whether a patient has heart disease from clinical measurements, and compares three classification models.

## Dataset
Cleveland heart disease dataset (UCI Machine Learning Repository): 303 patient records, 13 features and a binary target (`1` = heart disease present, `0` = absent). One duplicate row is removed during cleaning.

| Feature | Meaning |
|---|---|
| age, sex | Patient age and sex |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG result |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope, ca, thal | ST slope, number of major vessels, thalassemia result |

## Workflow
1. Load and inspect the data
2. Clean the data (missing values, duplicates)
3. Exploratory data analysis (class balance, distributions, correlations)
4. Stratified 80/20 train-test split
5. Train Logistic Regression, Decision Tree and Random Forest (Scikit-learn)
6. Evaluate with accuracy, precision, recall, F1, confusion matrix and 5-fold cross-validation
7. Inspect feature importance

## Results

| Model | Test Accuracy | Precision | Recall | F1 | 5-fold CV Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.787 | 0.763 | 0.879 | 0.817 | 0.838 |
| Decision Tree | 0.803 | 0.744 | 0.970 | 0.842 | 0.761 |
| Random Forest | 0.803 | 0.756 | 0.939 | 0.838 | 0.831 |

- With only ~300 records, the single test split (61 patients) is noisy, so cross-validated accuracy is the more reliable comparison. Logistic Regression and Random Forest are close (about 83-84%), and the Decision Tree is weaker (about 76%).
- For a medical screening task, **recall** matters because missing a patient with disease is costlier than a false alarm. Random Forest keeps high recall with better overall stability than a single tree.

Charts are in the `images/` folder and the full run is in the notebook.

## Limitations
Small, single-source dataset with no external validation. This is a learning project and **not** a medical diagnostic tool.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook
