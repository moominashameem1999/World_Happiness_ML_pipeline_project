import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

print("-----------------------TASK 6: Logistic Regression--------------------")
#creating binary label
master_median = Y.median()
Y_train_c = (Y_train > master_median).astype(int)
Y_test_c = (Y_test > master_median).astype(int)
X_train_c = X_train_scaled_df
X_test_c = X_test_scaled_df
lr = LogisticRegression(max_iter= 1000)
lr.fit(X_train_c, Y_train_c)
y_pred_lr = lr.predict(X_test_c)

print(f"Accuracy_lr : {accuracy_score(Y_test_c, y_pred_lr)}")
print(classification_report(Y_test_c, y_pred_lr))
sns.heatmap(confusion_matrix(Y_test_c, y_pred_lr), annot= True, fmt='d', cmap= 'Blues')
plt.title("Logistic Regression : Confusion Matrix")
plt.show()