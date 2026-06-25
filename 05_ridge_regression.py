import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

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

print("----------------------TASK 5: Ridge Regression-------------------------")
for alpha in [0.1, 1.0, 10.0, 100.0]:
    rr = Ridge(alpha= alpha)
    rr.fit(X_train_scaled, Y_train)
    Y_pred_rr = rr.predict(X_test_scaled)
    print(f"Alpha_rr : {alpha} -> R2: {r2_score(Y_test, Y_pred_rr):.4f}')")