import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("world_happiness_report.csv")
df = df.drop(columns=['Country', 'Happiness Rank', 'Standard Error', 'Dystopia Residual'])

le = LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])

X = df.drop(columns = ["Happiness Score"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("----------------------TASK 9: K-Means Clustering------------------------")
#step 1: Elbow method
inertia = []
k_range = range(2,11)
for k in k_range:
    km = KMeans(n_clusters= k, random_state= 42, n_init= 10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, marker = 'o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.show()

#step 2: final model with best k=3
km_final = KMeans(n_clusters= 3, random_state= 42, n_init= 10)
df["Clusters"] = km_final.fit_predict(X_scaled)
print("Silhouette score :", silhouette_score(X_scaled, df["Clusters"]))

#step 3: Visualize clusters (GDP Vs Happiness )
colors = ['red', 'blue', 'green']
for cluster in range(3):
    subset = df[df["Clusters"] == cluster]
    plt.scatter(subset["Economy (GDP per Capita)"], subset["Happiness Score"], label = f'cluster : {cluster}')
plt.xlabel("Economy (GDP per Capita)")
plt.ylabel("Happiness Score")
plt.title("K-Means Clusters")
plt.legend()
plt.savefig('cluster_plot.png')
plt.show()