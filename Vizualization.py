import logging
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import numpy as np
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud


# Configure logging to record errors in 'text_anal.log'
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def plot_ngrams(words, n, list_name):
    """
    Generates and saves a bar plot of the most common n-grams in a given list of words,
    with y-axis labels displayed in a table to the right of the plot.

    Parameters:
    words (list of str): The list of words from which to generate n-grams.
    n (int): The number of words in each n-gram.
    list_name (str): The name of the list, used to create the save path for the plot.

    Returns:
    None: The function saves the plot as a file and does not return any value.
    """
    try:
        # Ensure the directory exists
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)

        # Generate n-grams and calculate frequencies
        n_grams = list(ngrams(words, n))
        ngram_freq = Counter(n_grams)
        top_ngrams = ngram_freq.most_common(10)

        # Prepare plot labels and values
        labels, values = zip(*top_ngrams) if top_ngrams else ([], [])
        indexes = range(len(labels))

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(indexes, values, align='center')

        # Add a table at the right of the axes
        table_data = [[label] for label in labels]
        ax.table(cellText=table_data, colLabels=['N-grams'], cellLoc='center', loc='right')

        # Adjust layout to make room for the table:
        plt.subplots_adjust(right=0.5)
        plt.xticks(indexes, [''] * len(labels))  # Hide the x labels
        plt.tight_layout()

        # Remove axes spines and grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        
        # Save and close the plot
        plt.savefig(f'results/{list_name}/top_{n}_grams.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_ngrams for list {list_name}: {e}")

def create_wordcloud(word_freq, list_name):
    """
    Creates and saves a word cloud image based on the frequency distribution of words
    using an image silhouette as a mask.

    Parameters:
    word_freq (dict): A dictionary with words as keys and their frequencies as values.
    list_name (str): The name of the list, used to create the save path for the word cloud image.

    Returns:
    None: The function saves the word cloud image as a file and does not return any value.
    """
    try:
        # Ensure the 'results' directory exists
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)
        
        # Assuming the 'Images' directory is in the same directory as the script
        image_path = next(Path().cwd().joinpath('Images').glob('*.png'))  # Grabbing the first .png image

        # Load the silhouette for the mask
        silhouette_image = Image.open(image_path)
        mask = np.array(silhouette_image)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', mask=mask).generate_from_frequencies(word_freq)
        
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
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)

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
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)

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
