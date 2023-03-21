import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]+', ' ', text)
    text = text.lower()
    return text


# Load the dataset
data = pd.read_csv('dataset/product_reviews/original/amazon_reviews.txt', delimiter='\t')

# Preprocess the text
data['REVIEW_TEXT'] = data['REVIEW_TEXT'].apply(preprocess_text)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['REVIEW_TEXT'])

# Normalize the vectors
X_normalized = normalize(X)

# Perform Spectral Clustering using NCUT
n_clusters = 2  # Change this value according to the number of clusters you want to create
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, n_neighbors=10)
labels = spectral_clustering.fit_predict(X_normalized)

# Assign the cluster labels to the dataset
data['Cluster'] = labels

# Save the clustered data to a CSV file
data.to_csv('dataset/product_reviews/clustered_data.csv', index=False)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_normalized.toarray())

# Plot the clusters
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('viridis', n_clusters)
for i in range(n_clusters):
    cluster_points = X_embedded[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', cmap=cmap, edgecolors='k')
plt.legend()
plt.title('Clustered Data Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
