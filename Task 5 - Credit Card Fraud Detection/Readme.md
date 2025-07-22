# 💳 Task 5: Credit Card Fraud Detection - CodSoft Data Science Internship

This project is part of the **CodSoft Data Science Internship**. The objective was to build a machine learning model to detect **fraudulent transactions** from a highly imbalanced credit card dataset using classification techniques.

---

## 📌 Problem Statement

Build a binary classification model that accurately identifies whether a credit card transaction is:
- `0`: Genuine
- `1`: Fraudulent

---

## 📂 Dataset

- Source: Kaggle - Credit Card Fraud Detection Dataset  
- Records: 284,807 transactions  
- Features:  
  - `Time`: Seconds elapsed between each transaction and the first transaction  
  - `Amount`: Transaction amount  
  - `V1` to `V28`: PCA-transformed anonymized features  
  - `Class`: Target label (0 for genuine, 1 for fraud)

---

## ✅ Approach

1. **Data Preprocessing**
   - Scaled `Time` and `Amount` columns using `StandardScaler`
   - Checked and confirmed there were no missing values

2. **Handling Class Imbalance**
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to oversample the minority class (fraud)

3. **Modeling**
   - Split data into training and testing sets using `train_test_split`
   - Trained a **Random Forest Classifier**
   - Used `class_weight='balanced'` where necessary

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
   - Classification Report

---

## 🔍 Results

- ✅ **Accuracy**: `99.95%`  
- ✅ **Precision**: `98.64%`  
- ✅ **Recall**: `74.49%`  
- ✅ **F1-Score**: `84.88%`  
- ✅ **Confusion Matrix**:
[[56863 1]
[ 25 73]]

These metrics show the model's ability to detect fraudulent transactions with high confidence and recall, which is crucial in real-world fraud detection systems.

---

## 🛠 Tools & Libraries

- Python  
- Pandas  
- Scikit-learn  
- imbalanced-learn (SMOTE)  
- RandomForestClassifier  
- StandardScaler  
- Classification metrics

---

## 📁 Files

- `main.py` — Full pipeline from data loading to evaluation  
- `creditcard.csv` — Dataset (Because of file size [143 mb], I can't upload it here, you can see that dataset on kaggle)

---

## 🚀 Conclusion

This task was a valuable learning experience in handling **imbalanced datasets**, evaluating performance with **recall & F1-score**, and applying **real-world fraud detection logic** using Python and machine learning.

---

## 🔗 Links

- 📂 [GitHub Repository](https://github.com/touqeer-raza06/CODSOFT)
- 🌐 [LinkedIn Post](https://www.linkedin.com/posts/mohammed-touqeer-raza-344304331_creditcardfraud-frauddetection-machinelearning-activity-7353468592073375746-7DM9?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFOEUOQBwr88GAUekOiiQ6QdAq_Fz7v9ODI)


