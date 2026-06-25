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

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X.columns)

print("-----------------------TASK 2: Linear Regression--------------------")
#Task 2 :Linear Regression
X_train_g = X_train_scaled_df[["Economy (GDP per Capita)"]]
X_test_g = X_test_scaled_df[["Economy (GDP per Capita)"]]

slr = LinearRegression()
slr.fit(X_train_g, Y_train)
y_pred_slr = slr.predict(X_test_g)
slope = slr.coef_[0]
intercept = slr.intercept_

print("Slope_slr :", slope)
print("Intercept_slr :", intercept)
print("MSE_slr :", mean_squared_error(Y_test, y_pred_slr))
print("R2_slr: ", r2_score(Y_test, y_pred_slr))

#plotting the regression line
plt.scatter(X_test_g, Y_test, color = "blue")
plt.plot(X_test_g, y_pred_slr, color = "red")
plt.xlabel("Economy (GDP per Capita)")
plt.ylabel("Happiness score")
plt.title("Linear Regression")
plt.show()