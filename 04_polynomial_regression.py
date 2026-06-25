import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

X_train_g = X_train_scaled_df[["Economy (GDP per Capita)"]]
X_test_g = X_test_scaled_df[["Economy (GDP per Capita)"]]

print("---------------------TASK 4: Polynomial Regression-------------------")
poly = PolynomialFeatures(degree = 2)
X_poly_train = poly.fit_transform(X_train_g)
X_poly_test = poly.transform(X_test_g)
pr = LinearRegression()
pr.fit(X_poly_train, Y_train)
Y_poly_pred = pr.predict(X_poly_test)

print("R2_pr: ", r2_score(Y_test, Y_poly_pred))
print("MAE_pr: ", mean_absolute_error(Y_test, Y_poly_pred))
print('RMSE_pr:', np.sqrt(mean_squared_error(Y_test, Y_poly_pred)))