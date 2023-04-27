import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import time
import pickle


# Include the spectral clustering functions from previous answers

def load_data(filepath, n_sample=-1):
    data = pd.read_json(filepath, lines=True)
    # get the first 200 samples randomly, make sure is_deceptive contains both True and False
    # use do while loop to make sure we get both True and False
    test_run = False if n_sample == -1 else True
    while test_run:
        print(f"Trying to get {n_sample} samples, where is_deceptive contains 2 classes ...")
        data = data.sample(n=n_sample, random_state=42)
        if len(data['is_deceptive'].unique()) == 2:
            break

    return data


def load_datasets(dataset_path, n_sample=-1):
    dataset_names = [dir_name for dir_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, dir_name))]
    dataset_names.remove("results")
    dataset_names.remove("figures")
    dataset_names.remove("models")
    datasets = []
    print("Foud datasets...")
    print(dataset_names)
    for name in dataset_names:
        cleaned_filepath = os.path.join(dataset_path, name, 'cleaned', f'{name}.json')
        print(f"Loading {cleaned_filepath}")
        if os.path.isfile(cleaned_filepath):
            data = load_data(cleaned_filepath, n_sample)
            datasets.append((name, data))

    print(f"{cleaned_filepath} loaded !")
    return datasets

def plot_and_save_results(name, results, dataset_path):
    print(f"Plotting {name} results ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot([result['n_clusters'] for result in results], [result['silhouette'] for result in results], label='Silhouette Score')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.legend()

    ax2.plot([result['n_clusters'] for result in results], [result['calinski_harabasz'] for result in results], label='Calinski-Harabasz Index')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Calinski-Harabasz Index')
    ax2.legend()

    plt.suptitle(f'{name} - Clustering Evaluation Results')
    plt.tight_layout()

    fig_path = os.path.join(dataset_path, 'figures', 'ncut', name)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, 'results.png'))
    plt.close()


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_results_as_csv(name, results, dataset_path):
    df = pd.DataFrame(results)
    csv_path = os.path.join(dataset_path, 'results', 'ncut', f'{name}.csv')
    create_directory_if_not_exists(os.path.dirname(csv_path))
    
    # Calculate average, standard deviation, and range for Silhouette score and Calinski-Harabasz index
    df['silhouette_mean'] = df['silhouette'].mean()
    df['silhouette_std'] = df['silhouette'].std()
    df['silhouette_range'] = df['silhouette'].max() - df['silhouette'].min()
    
    df['calinski_harabasz_mean'] = df['calinski_harabasz'].mean()
    df['calinski_harabasz_std'] = df['calinski_harabasz'].std()
    df['calinski_harabasz_range'] = df['calinski_harabasz'].max() - df['calinski_harabasz'].min()

    df.to_csv(csv_path, index=False)


def main(args):
    dataset_path = args.dataset_path
    n_sample = args.n_sample
    datasets = load_datasets(dataset_path, n_sample)

    create_directory_if_not_exists(os.path.join(dataset_path, 'figures', 'ncut'))
    create_directory_if_not_exists(os.path.join(dataset_path, 'models', 'ncut'))

    for name, data in datasets:
        fig_path = os.path.join(dataset_path, 'figures', 'ncut', name)
        create_directory_if_not_exists(fig_path)

        model_path = os.path.join(dataset_path, 'models', 'ncut', name)
        create_directory_if_not_exists(model_path)

        X = data['text']
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_val = vectorizer.transform(X_val)

        MAX_CLUSTERS = 3
        n_clusters_range = list(range(2, MAX_CLUSTERS))
        results = []

        for n_clusters in n_clusters_range:
            start_time = time.time()
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, n_jobs=-1)
            cluster_labels_train = spectral.fit_predict(X_train)
            elapsed_time = time.time() - start_time

            cluster_labels_val = spectral.fit_predict(X_val)

            silhouette = silhouette_score(X_val, cluster_labels_val)
            calinski_harabasz = calinski_harabasz_score(X_val.toarray(), cluster_labels_val)

            results.append({'n_clusters': n_clusters, 'silhouette': silhouette, 'calinski_harabasz': calinski_harabasz, 'time': elapsed_time})

            print(f"Dataset: {name}, n_clusters: {n_clusters}, Time elapsed: {elapsed_time} seconds")

        save_results_as_csv(name, results, dataset_path)
        plot_and_save_results(name, results, dataset_path)

        best_result = max(results, key=lambda x: x['silhouette'])
        best_n_clusters = best_result['n_clusters']

        best_model = SpectralClustering(n_clusters=best_n_clusters, affinity='nearest_neighbors', n_neighbors=10, n_jobs=-1)
        best_model.fit(X_train)

        model_file = os.path.join(model_path, 'best_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dataset', help='Path to the directory containing datasets')
    parser.add_argument('--n_sample', '-n', type=int, default=-1, help='If > 0, only a small subset (of n_sample) of the data will be used')
    args = parser.parse_args()

    main(args)
