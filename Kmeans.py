# Import the modules and Read the data.
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Print the first five records
df = pd.read_csv("https://raw.githubusercontent.com/jiss-sngce/CO_3/main/jkcars.csv")
print(df.head())
# Get the total number of rows and columns, data types of columns and missing values (if exist) in the dataset.
print(df.shape)
print()
print(df.info())
# Create a new DataFrame consisting of three columns 'Volume', 'Weight', 'CO2'.
new_data = df[['Volume', 'Weight', 'CO2']]

# Print the first 5 rows of this new DataFrame.
print(new_data.head(5))
# Calculate inertia for different values of 'K'.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create an empty list to store silhouette scores obtained for each 'K'
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
    kmeans.fit(new_data)
    score = silhouette_score(new_data, kmeans.labels_)
    sil_scores.append(score)
silhoutte_df = pd.DataFrame({'K': range(2, 11), 'Silhouette Score': sil_scores})
print(silhoutte_df)
# Plot silhouette scores vs number of clusters.
plt.figure(figsize=(9, 5))
plt.plot(silhoutte_df['K'], silhoutte_df['Silhouette Score'], marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
# Clustering the dataset for K = 3
# Perform K-Means clustering with n_clusters = 3 and random_state = 10
k_means = KMeans(n_clusters=3, random_state=10)

# Fit the model to the scaled_df
k_means.fit(new_data)

# Make a series using predictions by K-Means
clusters = pd.Series(k_means.predict(new_data))

# Create a DataFrame with cluster labels for cluster visualisation
df['cluster'] = clusters
print(df.head())
