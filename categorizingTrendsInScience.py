# All the following code has been written by the IU University student Guglielmo Luraschi Sicca 
# Matriculation: 92125339, for the exam Categorizing-Trends-in-Science

# Start of the code

# Preliminary checks
# Check if the arXiv dataset has been downloaded

import os
import sys

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Prompt for arXiv dataset download
download_dataset = input("Have you already downloaded the arXiv dataset? (y/n): ").lower()

if download_dataset == "n":
    print("Please download the dataset from the following link: https://www.kaggle.com/datasets/Cornell-University/arxiv")
    print("Once downloaded, extract the contents to the same directory as this script.")
    print(f"The expected location after extraction should be: {script_dir}")
    print("After you have placed the dataset in the correct location, run this script again.\n")
    exit()

elif download_dataset != "y":
    print("Invalid input. Please enter 'y' or 'n'.")
    exit()

# If the dataset is downloaded, we assume it's in the script directory
# No need for the dataset_location prompt anymore

print("Then let's start categorizing trends in science! \n Please note that depending on the resources available to you it might take from a couple of minutes to an hour to run the program, \n in the meantime feel free to do something else, \n when the program has finished you will see a standard matplotlib pop up lighting up in in the taskbar showing you first the word cloud \n then, once you close it, a seaborn bar chart. \n Once closed those two windows the program will end.")

# Set path to the script directory
path = script_dir

# Check for expected data file
expected_data_file = "arxiv-metadata-oai-snapshot.json" 
data_file_path = os.path.join(path, expected_data_file)

if not os.path.isfile(data_file_path):
    print(f"Error: Expected data file '{expected_data_file}' not found in '{path}'.")
    print("Please ensure the dataset is correctly downloaded and extracted in the script's directory.")
    exit()
else:
    print(f"Found expected data file: {data_file_path}")

# Main program

import json
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import nltk
from wordcloud import WordCloud
from collections import Counter

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Create a Filtered JSON file (only 2024 data)
def create_filtered_json(input_file, output_file, target_year=2024):
    """Creates a new JSON file containing only entries from the specified year."""
    total_lines = 0
    written_lines = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_lines += 1
            try:
                doc = json.loads(line)
                latest_version_date = sorted(doc['versions'], key=lambda x: x['created'])[-1]['created']

                try:
                    date_obj = datetime.strptime(latest_version_date, "%a, %d %b %Y %H:%M:%S %Z")
                except ValueError:
                    try:
                        date_obj = datetime.strptime(latest_version_date, "%Y-%m-%d %H:%M:%S %Z")
                    except ValueError:
                        print(f"Skipping invalid date format: {latest_version_date}")
                        continue

                if date_obj.year == target_year:
                    json.dump(doc, outfile)
                    outfile.write('\n')
                    written_lines += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line content: {line.strip()}")
                continue
    print(f"Total lines processed: {total_lines}")
    print(f"Lines written to {output_file}: {written_lines}")

# Data Preprocessing
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
stop_words.update(['paper', 'study', 'research', 'abstract', 'conclusion', 'method',
                'result', 'use', 'using', 'used', 'model', 'system', 'graph',
                'algorithm', 'problem', 'data', 'performance', 'task', 'state',
                'group','space','function','algebra','prove','theory','set',
                'operator','number','approach', 'control', 'learning', 'training',
                'generation', 'feature', 'knowledge', 'prompt', 'architecture',
                'prediction', 'analysis', 'temperature', 'information', 'proposed',
                'challenge', 'point', 'series', 'equation', 'al'])
# Lemmatize stop words      
stop_words = set([lemmatizer.lemmatize(word) for word in stop_words])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    # Apply lemmatization before stop word removal
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Main Analysis Function
def analyze_trends_for_year(input_file, n_clusters=8, top_n_terms=50):
    """
    Analyzes the input JSON file to find trending topics for the year.
    """
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                doc = json.loads(line)
                data.append({
                    'id': doc.get('id', ''),
                    'created_date': sorted(doc['versions'], key=lambda x: x['created'])[-1]['created'],
                    'title': doc.get('title', ''),
                    'abstract': doc.get('abstract', '')
                })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line content: {line.strip()}")
                continue

    df = pd.DataFrame(data)
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df.dropna(subset=['created_date'], inplace=True)

    # Preprocess text
    df['processed_text'] = df['title'] + " " + df['abstract']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(2,3))
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=33)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    # Get top terms per cluster based on TF-IDF scores
    def get_top_terms_by_tfidf(cluster_id, tfidf_matrix, vectorizer, top_n):
        cluster_docs = df[df['cluster'] == cluster_id].index
        cluster_tfidf = tfidf_matrix[cluster_docs]
        avg_tfidf = cluster_tfidf.mean(axis=0).A1
        top_term_indices = avg_tfidf.argsort()[::-1][:top_n]
        return [(feature_names[i], avg_tfidf[i]) for i in top_term_indices]

    top_terms_per_cluster = {}
    for cluster_id in range(n_clusters):
        top_terms = get_top_terms_by_tfidf(cluster_id, tfidf_matrix, vectorizer, top_n_terms)
        top_terms_per_cluster[cluster_id] = top_terms

    # Flatten the list of top terms for word cloud and bar chart
    trending_words_with_scores = [term for terms in top_terms_per_cluster.values() for term in terms]
    trending_words, trending_scores = zip(*trending_words_with_scores)

    # Visualize Word Cloud and Bar Chart
    plot_wordcloud_and_barchart(trending_words_with_scores, top_n_terms)

    return top_terms_per_cluster

# Visualization Functions generate and display a word cloud and a bar chart from a list of words with scores
def plot_wordcloud_and_barchart(words_with_scores, top_n):

    # Word Cloud
    wordcloud_data = {word: score for word, score in words_with_scores}
    wordcloud = WordCloud(width=2000, height=1000, background_color='white').generate_from_frequencies(wordcloud_data)
    plt.figure(figsize=(28, 18))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Trending Words in science')
    plt.show()

    # Sort words and scores by score in descending order for bar chart
    sorted_words_with_scores = sorted(words_with_scores, key=lambda x: x[1], reverse=True)
    # Get the top N words and scores
    words, scores = zip(*sorted_words_with_scores[:top_n])  

    # Bar Chart
    plt.figure(figsize=(15, 15))
    sns.barplot(x=list(scores), y=list(words), palette="flare")
    plt.title('Top {} Trending Words in Science'.format(top_n))
    plt.xlabel('Average TF-IDF Score')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.show()

# Main Execution
create_filtered_json('arxiv-metadata-oai-snapshot.json', 'filtered_arxiv_data_2024.json')

top_terms_per_cluster = analyze_trends_for_year('filtered_arxiv_data_2024.json')