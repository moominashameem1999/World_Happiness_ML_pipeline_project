import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("world_happiness_report.csv")
df = df.drop(columns=['Country', 'Happiness Rank', 'Standard Error', 'Dystopia Residual'])

le = LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])

X = df.drop(columns = ["Happiness Score"])
Y = df["Happiness Score"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("------------------TASK 3: Multiple Linear Regression-----------------")
mlr = LinearRegression()
mlr.fit(X_train_scaled, Y_train)
Y_pred_mlr = mlr.predict(X_test_scaled)
slope = mlr.coef_[0]
intercept = mlr.intercept_

print("Slope_mlr :", slope)
print("Intercept_mlr :", intercept)
print("MSE_mlr :", mean_squared_error(Y_test, Y_pred_mlr))
print("R2_mlr: ", r2_score(Y_test, Y_pred_mlr))

plt.scatter(Y_test, Y_pred_mlr, color = "blue")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color = "red")
plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")
plt.title("Multiple Linear Regression")
plt.show()