import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Read dataset
df = pd.read_csv("D:\DSML Lab\datasets-main\jkcars.csv")
print(df.head())
print(df.info())

# Select numeric columns
new_data = df[['Volume', 'Weight', 'CO2']]
print(new_data.isnull().sum())

# Scale features
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)

# Find silhouette scores
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
    kmeans.fit(new_data_scaled)
    score = silhouette_score(new_data_scaled, kmeans.labels_)
    sil_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(9, 5))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Best K
optimal_k = sil_scores.index(max(sil_scores)) + 2
print("Optimal K:", optimal_k)

# Final clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=10, n_init=10)
df['cluster'] = kmeans.fit_predict(new_data_scaled)
print(df.head())

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(df['Weight'], df['CO2'], c=df['cluster'], cmap='rainbow')
plt.title(f'K-Means Clustering (K={optimal_k})')
plt.xlabel('Weight')
plt.ylabel('CO2')
plt.show()
