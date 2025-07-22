# 🌸 Task 3: Iris Flower Classification - CodSoft Data Science Internship

This project is part of the **CodSoft Data Science Internship**. The objective is to build a machine learning model that classifies iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on the measurements of their petals and sepals.

---

## 📌 Problem Statement

Build a classifier using the **Iris Dataset** to identify the species of an iris flower based on the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## 📂 Dataset

- The dataset contains **150 samples** and **5 columns** (4 features + 1 target label).
- Each sample is labeled as one of the three species:
  - `Iris-setosa`
  - `Iris-versicolor`
  - `Iris-virginica`

---

## ✅ Approach

1. **Import Libraries & Load Data**
2. **Data Preprocessing**  
   - Checked for null values  
   - No encoding needed as labels are already text-based
3. **Feature and Target Split**
4. **Train-Test Split**  
   - 80% training  
   - 20% testing
5. **Model Used:**  
   - `RandomForestClassifier` (from `sklearn.ensemble`)
6. **Evaluation Metrics:**  
   - Accuracy Score

---

## 🔍 Results

- Achieved an **accuracy of ~96–100%** on the test set depending on random state.
- The model performs well on all three classes due to balanced and clean data.

---

## 🧠 Tools & Libraries

- Python  
- Pandas  
- Scikit-learn  
- RandomForestClassifier  

---


## 📁 Files

- `main.py` — contains the code
- `IRIS.csv` — dataset file
- `output.png` — sample output screenshot

---

## 🚀 Conclusion

This was a beginner-friendly classification task that offered hands-on experience with supervised learning, specifically with the widely used **Iris Dataset**. Great exercise to understand how machine learning models handle multi-class classification problems.

