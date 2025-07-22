import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, classification_report

df = pd.read_csv("creditcard.csv")

x = df.drop(["Class"], axis=1)
y = df["Class"]

scaler = StandardScaler()
x[["Time", "Amount"]] = scaler.fit_transform(x[["Time", "Amount"]])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight='balanced')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy : ", accuracy_score(y_test, y_pred)*100, "%")
print("Recall Score : ", recall_score(y_test, y_pred)*100, "%")
print("Precision Score : ", precision_score(y_test, y_pred)*100, "%")
print("F1 Score : ", f1_score(y_test, y_pred)*100, "%")
print("** Confuison Matrix **\n", confusion_matrix(y_test, y_pred))
print("** Classification Report **\n", classification_report(y_test, y_pred))
