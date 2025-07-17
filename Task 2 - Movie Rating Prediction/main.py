import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')
df = df.drop(["Name", "Director", "Actor 1", "Actor 2", "Actor 3"], axis=1)

df = df.dropna(subset=["Genre", "Rating"])

#for duration
df["Duration"] = df["Duration"].str.replace(" min", "").astype(float)
df["Duration"] = df["Duration"].fillna(df["Duration"].mean())

#for votes
df["Votes"] = df["Votes"].fillna(0)
df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors='coerce').fillna(0)

#for year
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Year"] = df["Year"].fillna(df["Year"].mean()).astype(int)

#for genre
df["Genre"] = df["Genre"].str.split(", ")
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df["Genre"]), columns=mlb.classes_)
df = pd.concat([df.drop("Genre", axis=1), genre_dummies], axis=1)

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])

x=df.drop("Rating", axis=1)
y=df["Rating"]

#training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#actual training
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor().fit(x_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(x_test)
print("Mean Squared Error : ", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error : ", mean_absolute_error(y_test, y_pred))
print("R2 Score : ", r2_score(y_test, y_pred), "or ", r2_score(y_test, y_pred)*100, "%")

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted IMDb Ratings")
plt.grid(True)
plt.show()