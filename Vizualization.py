import logging
import os
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud

# Configure logging to record errors in 'text_anal.log'
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def plot_ngrams(words, n, list_name):
    """
    Generates and saves a bar plot of the most common n-grams in a given list of words.

    Parameters:
    words (list of str): The list of words from which to generate n-grams.
    n (int): The number of words in each n-gram.
    list_name (str): The name of the list, used to create the save path for the plot.

    Returns:
    None: The function saves the plot as a file and does not return any value.
    """
    try:
        # Ensure the directory exists
        os.makedirs(f'results/{list_name}', exist_ok=True)

        # Generate n-grams and calculate frequencies
        n_grams = list(ngrams(words, n))
        ngram_freq = Counter(n_grams)
        top_ngrams = ngram_freq.most_common(10)

        # Prepare plot labels and values
        labels, values = zip(*top_ngrams) if top_ngrams else ([], [])
        indexes = range(len(labels))

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.bar(indexes, values, align='center')
        plt.xticks(indexes, [' '.join(label) for label in labels], rotation='vertical')
        plt.tight_layout()

        # Save and close the plot
        plt.savefig(f'results/{list_name}/top_{n}_grams.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_ngrams for list {list_name}: {e}")

def create_wordcloud(word_freq, list_name):
    """
    Creates and saves a word cloud image based on the frequency distribution of words.

    Parameters:
    word_freq (dict): A dictionary with words as keys and their frequencies as values.
    list_name (str): The name of the list, used to create the save path for the word cloud image.

    Returns:
    None: The function saves the word cloud image as a file and does not return any value.
    """
    try:
        # Ensure the directory exists
        os.makedirs(f'results/{list_name}', exist_ok=True)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Save and close the plot
        plt.savefig(f'results/{list_name}/wordcloud.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in create_wordcloud for list {list_name}: {e}")

def plot_word_network(words, list_name):
    """
    Creates and saves a network graph of words based on their bigram relationships.

    Parameters:
    words (list of str): A list of words to be used for creating the network graph.
    list_name (str): The name of the list, used to create the save path for the network graph image.

    Returns:
    None: The function saves the network graph as a file and does not return any value.
    """
    try:
        # Ensure the directory exists
        os.makedirs(f'results/{list_name}', exist_ok=True)

        # Generate network graph
        G = nx.Graph()
        G.add_edges_from(ngrams(words, 2))

        # Plotting
        plt.figure(figsize=(12, 8))
        nx.draw(G, with_labels=True, font_weight='bold')

        # Save and close the plot
        plt.savefig(f'results/{list_name}/word_network.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_word_network for list {list_name}: {e}")

def plot_lexical_diversity_histogram(diversities, list_name):
    """
    Creates and saves a histogram of lexical diversities.

    Parameters:
    diversities (list of float): A list of lexical diversity scores to be plotted.
    list_name (str): The name of the list, used to create the save path for the histogram.

    Returns:
    None: The function saves the histogram as a file and does not return any value.
    """
    try:
        # Ensure the directory exists
        os.makedirs(f'results/{list_name}', exist_ok=True)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.hist(diversities, bins=20, color='skyblue')
        plt.title('Lexical Diversity Histogram')
        plt.xlabel('Lexical Diversity')
        plt.ylabel('Frequency')

        # Save and close the plot
        plt.savefig(f'results/{list_name}/lexical_diversity_histogram.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_lexical_diversity_histogram for list {list_name}: {e}")
