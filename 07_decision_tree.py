import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

print("-----------------------TASK 7: Decision Tree--------------------------")
dt = DecisionTreeClassifier(max_depth=4, random_state= 42)
dt.fit(X_train_c, Y_train_c)
Y_pred_dt = dt.predict(X_test_c)
print("Accuracy_dt :", accuracy_score(Y_test_c, Y_pred_dt))
print(classification_report(Y_test_c, Y_pred_dt))
#Visualizing the Tree
plt.figure(figsize=(18, 8))
feature_names = X.columns.to_list()
plot_tree(dt, feature_names= feature_names, class_names= ["Not Happy", "Happy"], filled= True, rounded= True)
plt.title("Decision Tree")
plt.show()