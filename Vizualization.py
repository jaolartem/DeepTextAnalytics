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
        # Crear el directorio si no existe
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)

        # Generar n-grams y calcular frecuencias
        n_grams = list(ngrams(words, n))
        ngram_freq = Counter(n_grams)
        top_ngrams = ngram_freq.most_common(10)

        if not top_ngrams:
            logging.warning(f"No hay suficientes n-grams para {list_name}")
            return  # Salir si no hay n-grams

        # Preparar etiquetas y valores para la gráfica
        labels, values = zip(*top_ngrams)
        indexes = range(len(labels))
        colors = plt.cm.jet(np.linspace(0, 1, len(labels)))

        # Crear y guardar la gráfica
        fig, ax = plt.subplots(figsize=(24, 16))
        ax.bar(indexes, values, align='center', color=colors)
        ax.set_xticks(indexes)
        ax.set_xticklabels([' '.join(ngram) for ngram in labels], rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'results/{list_name}/top_{n}_grams.png')
        plt.close()

    except Exception as e:
        logging.error(f"Error en plot_ngrams para {list_name}: {e}")


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
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)
        
        images_path = Path().cwd().joinpath('Images')
        if not images_path.exists() or not any(images_path.glob('*.png')):
            logging.error(f"No hay imágenes en {images_path}")
            return  # Salir si no hay imágenes

        image_path = next(images_path.glob('*.png'))
        silhouette_image = Image.open(image_path)
        mask = np.array(silhouette_image)
        wordcloud = WordCloud(width=800, height=400, background_color='white', mask=mask).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f'results/{list_name}/wordcloud.png')
        plt.close()

    except Exception as e:
        logging.error(f"Error en create_wordcloud para {list_name}: {e}")

        

def plot_word_network(words, list_name):
    """
    Creates and saves a network graph of words based on their bigram relationships,
    with a black background and neon colors, considering only single-word nodes,
    and limited to the first 50 words.

    Parameters:
    words (list of str): A list of words to be used for creating the network graph.
    list_name (str): The name of the list, used to create the save path for the network graph image.

    Returns:
    None: The function saves the network graph as a file and does not return any value.
    """

def plot_word_network(words, list_name):
    try:
        Path(f'results/{list_name}').mkdir(parents=True, exist_ok=True)
        words = words[:50]

        G = nx.Graph()
        G.add_nodes_from(words)

        edges = [(words[i], words[i+1]) for i in range(len(words) - 1)]
        G.add_edges_from(edges)

        colors = plt.cm.spring(np.linspace(0, 1, len(G.nodes)))

        plt.figure(figsize=(15, 10), facecolor='k')
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw_networkx(G, pos, node_color=colors, with_labels=True, font_color='white')
        plt.savefig(f'results/{list_name}/word_network.png', facecolor='k')
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
        plt.savefig(f'results/{list_name}/lexical_diversity_histogram.png')
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
        plt.savefig(f'results/{list_name}/word_length_distribution.png')
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
        plt.savefig(f'results/{list_name}/word_co_occurrence_network.png')
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
        plt.savefig(f'results/{list_name}/readability_index_chart.png')
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
        plt.savefig(f'results/{list_name}/pos_frequency_distribution.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_pos_frequency_distribution for list {list_name}: {e}")
