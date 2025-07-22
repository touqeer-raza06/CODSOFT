import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("advertising.csv")

x = df.drop(["Sales"], axis=1)
y = df["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("R2 Score : ", r2_score(y_test, y_pred)*100, "%")
print("Mean Squared Error : ", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error : ", mean_absolute_error(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

feat_importances = pd.Series(model.feature_importances_, index=x.columns)
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Feature Importance")
plt.show()