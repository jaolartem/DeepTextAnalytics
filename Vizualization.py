import logging
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import numpy as np
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud
from random import choice
import glob
import random



def neon_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    neon_colors = ["#94fc13", "#13fcf2", "#f213fc", "#fc138f", "#fc8313"]
    return random.choice(neon_colors)


def plot_ngrams(words, n, list_name):
    """
    Generates and saves a bar plot of the most common n-grams in a given list of words,
    with enlarged y-axis labels and soft pastel colors. Each n-gram name in the legend
    is accompanied by a color box corresponding to its bar in the plot.

    Parameters:
    words (list of str): The list of words from which to generate n-grams.
    n (int): The number of words in each n-gram.
    list_name (str): The name of the list, used to create the save path for the plot.

    Returns:
    None: The function saves the plot as a file and does not return any value.
    """
    try:
        # Create the directory if it does not exist
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)

        # Generate n-grams and calculate frequencies
        n_grams = list(ngrams(words, n))
        ngram_freq = Counter(n_grams)
        top_ngrams = ngram_freq.most_common(10)
        if not top_ngrams:
            logging.warning(f"Not enough n-grams for {list_name}")
            return  # Exit if there are no n-grams

        # Prepare labels and values for the plot
        labels, values = zip(*top_ngrams)
        indexes = range(len(labels))
        
        # Using pastel colors
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))

        # Create and save the plot
        fig, ax = plt.subplots(figsize=(24, 16))
        bars = ax.bar(indexes, values, align='center', color=colors)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add Y-axis label and display values
        ax.set_ylabel('Frequency', fontsize=20) # Enlarged font
        ax.set_yticks(values)

        # Remove X-axis labels
        ax.set_xticks([])

        # Create custom legend with color boxes and enlarged font, remove border box
        legend_labels = [' '.join(ngram) for ngram in labels]
        legend = plt.legend(bars, legend_labels, loc='upper right', bbox_to_anchor=(1, 1), handletextpad=0.1, fontsize=23)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_facecolor('none')

        plt.tight_layout()
        plt.savefig(f'results/{list_name}/top_{n}_grams.png')
        plt.close()

    except Exception as e:
        logging.error(f"Error in plot_ngrams for {list_name}: {e}")




def create_wordcloud(word_freq, list_name):
    """
    Creates and saves a word cloud image based on the frequency distribution of words
    using a randomly selected image silhouette as a mask.

    Parameters:
    word_freq (dict): A dictionary with words as keys and their frequencies as values.
    list_name (str): The name of the list, used to create the save path for the word cloud image.

    Returns:
    None: The function saves the word cloud image as a file and does not return any value.
    """

    try:
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)
        
        images_path = Path().cwd().joinpath('Images')
        image_files = list(glob.glob(str(images_path / '*.png')))

        if not image_files:
            raise FileNotFoundError(f"No images found in {images_path}") 
        
        image_path = choice(image_files)  
        
        
        silhouette_image = Image.open(image_path)
        mask = np.array(silhouette_image)
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            mask= mask).generate_from_frequencies(word_freq)

        
        plt.figure(figsize=(40, 20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f'results/{list_name}/wordcloud.png')
        plt.close()

    except Exception as e:
        logging.error(f"Error en create_wordcloud para {list_name}: {e}")

        
def plot_word_network(words, list_name):
    try:
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)
        words = words[:100]

        G = nx.Graph()
        G.add_nodes_from(words)

        edges = [(words[i], words[i+1]) for i in range(len(words) - 1)]
        G.add_edges_from(edges)

        neon_colors = ["#94fc13", "#13fcf2", "#f213fc", "#fc138f", "#fc8313"]
        node_colors = [random.choice(neon_colors) for _ in G.nodes()]
        
        degree_centrality = nx.degree_centrality(G)

       
        node_sizes = [50000 * degree_centrality[node] for node in G.nodes()]    
        
        plt.figure(figsize=(32, 24))
        
        pos = nx.spring_layout(G, k=0.15, iterations=20, scale=900.0)
        
        
        nx.draw_networkx(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True, font_size=18)
        
        plt.axis('off')
        
        plt.savefig(f'results/{list_name}/word_network.png')
        plt.close()

    except Exception as e:
        logging.error(f"Error en plot_word_network para {list_name}: {e}")

 
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
        plt.savefig(f'results/lexical_diversity_histogram.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_lexical_diversity_histogram for list {list_name}: {e}")

def plot_word_length_distribution(word_lengths, list_name):
    """
    Generates and saves a histogram of word lengths.

    Args:
        word_lengths (dict): Dictionary with word lengths as keys and their frequencies as values.
        list_name (str): Name used for the save path of the histogram.

    Returns:
        None: Saves the histogram as a file, does not return a value.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(word_lengths.keys(), word_lengths.values(), color='skyblue')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.title('Word Length Distribution')
        plt.savefig(f'results/word_length_distribution.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_word_length_distribution for list {list_name}: {e}")


def plot_word_co_occurrence(co_occurrences, list_name):
    """
    Generates and saves a network graph of word co-occurrences.

    Args:
        co_occurrences (dict): Dictionary with word pairs (tuples) as keys and their co-occurrence frequencies as values.
        list_name (str): Name used for the save path of the graph image.

    Returns:
        None: Saves the network graph as a file, does not return a value.
    """
    try:
        G = nx.Graph()
        for pair, weight in co_occurrences.items():
            G.add_edge(pair[0], pair[1], weight=weight)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
                width=[v['weight'] for (u, v) in G.edges(data=True)])
        plt.title('Word Co-Occurrence Network')
        plt.savefig(f'results/word_co_occurrence_network.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_word_co_occurrence for list {list_name}: {e}")

def plot_readability_index(readability_scores, list_name):
    """
    Generates and saves a bar chart of readability scores for different texts.

    Args:
        readability_scores (dict): Dictionary with text identifiers as keys and Flesch Reading Ease scores as values.
        list_name (str): Name used for the save path of the readability chart.

    Returns:
        None: Saves the readability chart as a file, does not return a value.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(readability_scores)), readability_scores, align='center', color='teal')
        plt.xlabel('Documents')
        plt.ylabel('Readability Score')
        plt.title('Comparison of Readability Scores')
        plt.savefig(f'results/readability_index_chart.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_readability_index for list {list_name}: {e}")


def plot_pos_frequency_distribution(pos_frequencies, list_name):
    """
    Generates and saves a bar chart of POS tag frequencies.

    Args:
        pos_frequencies (Counter or dict): Dictionary or Counter object with POS tags as keys and their frequencies as values.
        list_name (str): Name used for the save path of the POS frequency chart.

    Returns:
        None: Saves the POS frequency chart as a file, does not return a value.
    """
    try:
        # Preparing data for plotting
        labels, values = zip(*pos_frequencies.items()) if pos_frequencies else ([], [])
        indexes = range(len(labels))

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(indexes, values, align='center', color='blue')
        plt.xticks(indexes, labels, rotation='vertical')
        plt.xlabel('POS Tags')
        plt.ylabel('Frequency')
        plt.title('POS Tag Frequency Distribution')

        # Save and close the plot
        plt.tight_layout()
        plt.savefig(f'results/pos_frequency_distribution.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_pos_frequency_distribution for list {list_name}: {e}")

def create_wordcloud_multi(word_freq, list_name):
    """
    Creates and saves a word cloud image from word frequencies, using a silhouette image as a mask.

    Parameters:
        word_freq (dict): Dictionary with word frequencies.
        list_name (str): Name used for the output file path.

    Returns:
        None: Saves the word cloud image without returning any value.
    """
    try:
        # Validate word frequency dictionary
        if not word_freq or not isinstance(word_freq, dict):
            raise ValueError("word_freq must be a dictionary of word frequencies.")

        # Create the output directory if it doesn't exist
        output_dir = Path(f'results/{list_name}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_path = Path().cwd().joinpath('Images')
        image_files = list(glob.glob(str(images_path / '*.png')))

        if not image_files:
            raise FileNotFoundError(f"No images found in {images_path}")
        
        image_path = choice(image_files)
        mask = np.array(Image.open(image_path))        
        
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            mask= mask).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(40, 20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_dir / 'wordcloud.png')
        plt.close()
        
    except ValueError as e:
        logging.error(f"Validation Error in create_wordcloud_multi for '{list_name}': {e}")
    except FileNotFoundError as e:
        logging.error(f"File Not Found Error in create_wordcloud_multi for '{list_name}': {e}")
    except Exception as e:
        logging.error(f"General Error in create_wordcloud_multi for '{list_name}': {e}")
