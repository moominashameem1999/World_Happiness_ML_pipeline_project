import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

master_median = Y.median()
Y_train_c = (Y_train > master_median).astype(int)
Y_test_c = (Y_test > master_median).astype(int)
X_train_c = X_train_scaled_df
X_test_c = X_test_scaled_df
feature_names = X.columns.to_list()

print("------------------------TASK 8: Random forest--------------------------")
rf = RandomForestClassifier(n_estimators= 100, random_state= 42)
rf.fit(X_train_c, Y_train_c)
y_pred_rf = rf.predict(X_test_c)
print("Accuracy_rf :", accuracy_score(Y_test_c, y_pred_rf))
print(classification_report(Y_test_c, y_pred_rf))
#feature importance
importances = pd.Series(rf.feature_importances_, index = feature_names)
importances.sort_values().plot(kind= "barh", color= "steelblue")
plt.title("Feature Importance - Random Forest")
plt.show()