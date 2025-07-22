# 📈 Task 4: Sales Prediction Using Python - CodSoft Data Science Internship

This project is part of the **CodSoft Data Science Internship**. The goal was to predict product sales based on advertising budgets across different media platforms using supervised machine learning techniques.

---

## 📌 Problem Statement

Build a regression model that predicts **sales** based on how much is spent on:
- **TV advertising**
- **Radio advertising**
- **Newspaper advertising**

---

## 📂 Dataset

- The dataset contains **200 rows** and the following columns:
  - `TV`: Advertising budget spent on TV (in thousands of dollars)
  - `Radio`: Advertising budget spent on Radio
  - `Newspaper`: Advertising budget spent on Newspaper
  - `Sales`: Units sold (Target variable)

---

## ✅ Approach

1. **Imported libraries & loaded dataset**
2. **Explored and cleaned data**
   - Checked for null values
   - No missing data found
3. **Split data into features (`TV`, `Radio`, `Newspaper`) and target (`Sales`)**
4. **Train-test split (80/20)**
5. **Model Used:**
   - `RandomForestRegressor` (from `sklearn.ensemble`)
6. **Evaluation Metrics:**
   - R² Score
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)

---

## 🔍 Results

- **R² Score**: `95.12%`
- **Mean Squared Error**: `1.50`
- **Mean Absolute Error**: `0.90`

The model shows excellent predictive performance, capturing most of the variability in sales based on ad spending.

---

## 🧠 Tools & Libraries

- Python  
- Pandas  
- Scikit-learn  
- RandomForestRegressor  

---


## 📁 Files

- `main.py` — Python code for training and evaluation
- `advertising.csv` — Dataset used for training
- `Figure 1.png` — Screenshot of model output
- `Figure 2.png` — Screenshot of model output

---

## 🚀 Conclusion

This task provided practical experience in solving a **regression problem**, emphasizing the relationship between marketing spend and product sales. A great real-world application of machine learning in business forecasting.

---

## 🔗 Links

- 📂 [GitHub Repository](https://github.com/touqeer-raza06/CODSOFT/tree/main/Task%202%20-%20Movie%20Rating%20Prediction)
- 🌐 [LinkedIn Post](---

## 🔗 Links

- 📂 [GitHub Repository](https://github.com/touqeer-raza06/CODSOFT)
- 🌐 [LinkedIn Post](https://www.linkedin.com/posts/mohammed-touqeer-raza-344304331_datascience-machinelearning-python-activity-7350486729297989632-EZpt?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFOEUOQBwr88GAUekOiiQ6QdAq_Fz7v9ODI)

)


