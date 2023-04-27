import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pickle
import time
import argparse

def load_model_and_dataset(model_path, dataset_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    dataset = pd.read_json(dataset_path, lines=True)
    return model, dataset

def load_n_samples(dataset, n_samples):
    dataset = dataset.sample(n=n_samples, random_state=42)
    dataset.reset_index(drop=True, inplace=True)

    return dataset
def evaluate_model(model, X, y, n_splits=5, n_samples=None):
    skf = StratifiedKFold(n_splits=n_splits)
    results = []

    if n_samples:
        dataset = load_n_samples(pd.concat([X, y], axis=1), n_samples)
        X = dataset['text']
        y = dataset['is_deceptive']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

    return pd.DataFrame(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the best model file')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset file')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to use for evaluation (default: use all)')
    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    n_samples = args.n_samples

    model, dataset = load_model_and_dataset(model_path, dataset_path)
    X = dataset['text']
    y = dataset['is_deceptive']

    start_time = time.time()
    results_df = evaluate_model(model, X, y, n_splits=5, n_samples=n_samples)

    print("Results for each fold:")
    print(results_df)

    results_df.to_csv('results.csv', index=False)
    # 4. Draw plots
    print("Plotting results...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    ax1.plot(range(1, 6), results_df['accuracy'], label='Accuracy')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(range(1, 6), results_df['precision'], label='Precision')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Precision')
    ax2.legend()

    ax3.plot(range(1, 6), results_df['recall'], label='Recall')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Recall')
    ax3.legend()

    ax4.plot(range(1, 6), results_df['f1'], label='F1-score')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('F1-score')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('results_plots.png')
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

