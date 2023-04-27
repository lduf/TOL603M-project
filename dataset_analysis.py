import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from concurrent.futures import ProcessPoolExecutor


nltk.download('stopwords')



def load_data(filepath):
    data = pd.read_json(filepath, lines=True)
    # keep only the first 100 samples
    return data

def analyze_dataset(name, data):
    name = name+'_final'
    os.makedirs(f'analyses/{name}', exist_ok=True)

    # Calculating the number of samples in each dataset
    n_samples = len(data)
    print(f'{name} dataset has {n_samples} samples.')

    # Checking for missing values
    missing_values = data.isnull().sum().sum()
    print(f'{name} dataset has {missing_values} samples with missing values.')

    # Analyzing the distribution of 'is_deceptive'
    deceptive_counts = data['is_deceptive'].value_counts().to_dict()
    print(f'{name} dataset deceptive distribution: {deceptive_counts}')

    # Investigating the distribution of text lengths
    text_lengths = data['text'].str.len()

    return n_samples, missing_values, deceptive_counts, text_lengths

def plot_statistics(name, n_samples, missing_values, deceptive_counts, text_lengths):
    name = name+'_final'
    sns.set(style='whitegrid')

    # Bar plots for number of samples and 'is_deceptive' distribution
    fig, ax = plt.subplots()
    sns.barplot(x=list(deceptive_counts.keys()), y=list(deceptive_counts.values()), ax=ax)
    ax.set_title(f'{name} dataset: Number of samples and deceptive distribution')
    fig.savefig(f'analyses/{name}/deceptive_distribution.png')

    # Histogram for text lengths
    fig, ax = plt.subplots()
    sns.histplot(text_lengths, ax=ax)
    ax.set_title(f'{name} dataset: Text length distribution')
    fig.savefig(f'analyses/{name}/text_length_distribution.png')

def create_wordclouds(name, data):
    text_all = ' '.join(data['text'])
    text_true = ' '.join(data[data['is_deceptive'] == True]['text'])
    text_false = ' '.join(data[data['is_deceptive'] == False]['text'])

    for label, text in [('all', text_all), ('true', text_true), ('false', text_false)]:
       # wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=None).generate(text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f'analyses/{name}/wordcloud_{label}.png')


def plot_most_used_words(name, data):
    name = name+'_final'
    data_true = data[data['is_deceptive'] == True]
    data_false = data[data['is_deceptive'] == False]

    stop_words = set(stopwords.words('english'))
    # also remove punctuation
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '$', '-', '*', '/', '\\', '>', '<', '=', '+', '%', '&', '|', '~', '`', '@', '#', '^', '_', '``', "''"])


    def count_words(texts):
        words = [word.lower() for text in texts for word in word_tokenize(text) if word.lower() not in stop_words]
        if len(words) == 0:
            return []
        return Counter(words).most_common(25)

    def plot_word_counts(counts, title, filename):
        if len(counts) == 0:
            return
        words, frequencies = zip(*counts)
        plt.figure(figsize=(16, 8))
        sns.barplot(x=list(words), y=list(frequencies))
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.25)  # Add extra space at the bottom
        plt.savefig(f'analyses/{name}/{filename}.png')
        #plt.show()

    all_counts = count_words(data['text'])
    true_counts = count_words(data_true['text'])
    false_counts = count_words(data_false['text'])

    plot_word_counts(all_counts, f'{name} - 25 Most Used Words (All)', '25_most_used_words_all')
    plot_word_counts(true_counts, f'{name} - 25 Most Used Words (Deceptive=True)', '25_most_used_words_true')
    plot_word_counts(false_counts, f'{name} - 25 Most Used Words (Deceptive=False)', '25_most_used_words_false')


def plot_boxplot(name, text_lengths):
    name = name+'_final'
    sns.set(style='whitegrid')
    fig, ax = plt.subplots()
    sns.boxplot(x=text_lengths, ax=ax)
    ax.set_title(f'{name} dataset: Text length boxplot')
    fig.savefig(f'analyses/{name}/text_length_boxplot.png')

def compare_datasets(statistics):
    summary = pd.DataFrame(statistics).T
    summary.columns = ['Number of samples', 'Missing values', 'Deceptive distribution', 'Text lengths']
    summary.to_csv('analyses/summary.csv', index_label='Dataset')

    print("\nSummary of all datasets:")
    print(summary)

def plot_tsne(name, data, perplexity=30, n_components=2, random_state=42):
    name = name+'_final'
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(data['text'])
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init='random')
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['is_deceptive'] = data['is_deceptive']

    fig, ax = plt.subplots()
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='is_deceptive', data=df_tsne, ax=ax)
    ax.set_title(f'{name} dataset: t-SNE plot with perplexity {perplexity}')
    fig.savefig(f'analyses/{name}/tsne_plot.png')


def process_dataset(name, dataset_path):
    cleaned_filepath = os.path.join(dataset_path, name, 'cleaned', f'{name}.json')
    if os.path.isfile(cleaned_filepath):
        data = load_data(cleaned_filepath)
        n_samples, missing_values, deceptive_counts, text_lengths = analyze_dataset(name, data)
        statistics = [n_samples, missing_values, deceptive_counts, text_lengths]
        plot_statistics(name, n_samples, missing_values, deceptive_counts, text_lengths)
        plot_boxplot(name, text_lengths)
        plot_tsne(name, data)
        plot_most_used_words(name, data)
        #create_wordclouds(name, data)
        return name, statistics
    else:
        print(f"Cleaned dataset file not found for {name}.")
        return name, None

def main(args):
    dataset_names = [dir_name for dir_name in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, dir_name))]

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_dataset, dataset_names, [args.dataset_path] * len(dataset_names))

    statistics = {name: stats for name, stats in results if stats is not None}
    compare_datasets(statistics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dataset', help='Path to the datasets directory')
    args = parser.parse_args()

    main(args)
