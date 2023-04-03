import pandas as pd
import re
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]+', ' ', text)
    text = text.lower()
    return text



# Load the dataset (assuming it's in JSON format)
data = pd.read_json('dataset/phishing/cleaned/phishing.json', lines=True)

#Preprocess the text
data['text'] = data['text'].apply(preprocess_text)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['text'])

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=3, random_state=42)
X_embedded = tsne.fit_transform(X.toarray())

# Add t-SNE components to the dataset
data['t-SNE Component 1'] = X_embedded[:, 0]
data['t-SNE Component 2'] = X_embedded[:, 1]
data['t-SNE Component 3'] = X_embedded[:, 2]

# Convert boolean labels to string
data['is_deceptive'] = data['is_deceptive'].apply(lambda x: 'Deceptive' if x else 'Non-deceptive')

# Create an interactive 3D scatter plot using Plotly
fig = px.scatter_3d(data, x='t-SNE Component 1', y='t-SNE Component 2', z='t-SNE Component 3',
                    color='is_deceptive', labels={'is_deceptive': 'Deception'},
                    title='Deceptive vs Non-deceptive Text Visualization')

fig.show()
# save the plot as an HTML file
fig.write_html('dataset/phishing/visualization.html')
