import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load data and preprocess independently
df = pd.read_csv("world_happiness_report.csv")
df = df.drop(columns=['Country', 'Happiness Rank', 'Standard Error', 'Dystopia Residual'])

le = LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])

print("----------------------------TASK 1: EDA-----------------------------")
print("Shape :", df.shape)
print(df.describe())
print("\nIsnull count:")
print(df.isnull().sum())

# 1. Histogram
df.hist(bins=20, figsize=(14, 10))
plt.tight_layout()
plt.title("Histogram Distributions")
plt.show()

# 2. Correlation Heatmap (Updated to 'Blues' per PDF instructions)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='inferno')
plt.title('Correlation Heatmap')
plt.show()

# 3. Boxplot for Outliers
df.boxplot(figsize=(14, 6))
plt.xticks(rotation=45)
plt.show()