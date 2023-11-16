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
       # Generate n-grams and calculate frequencies
        n_grams = list(ngrams(words, n))
        ngram_freq = Counter(n_grams)
        top_ngrams = ngram_freq.most_common(10)

        # Prepare plot labels and values
        labels, values = zip(*top_ngrams) if top_ngrams else ([], [])
        indexes = range(len(labels))

        # Define a unique color for each n-gram
        colors = plt.cm.jet([i / float(len(labels) - 1) for i in range(len(labels))])

        # Plotting
        fig, ax = plt.subplots(figsize=(24, 16))
        bars = ax.bar(indexes, values, align='center', color=colors)

        # Adding the numbered labels on top of each bar
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(i+1), 
                    ha='center', va='bottom')

        # Create a legend for the colors
        ngram_labels = [' '.join(ngram) for ngram in labels]
        plt.legend(bars, ngram_labels, title="N-grams")

        # Adjust layout
        plt.xticks(indexes, ngram_labels, rotation=45, ha='right')  # Show x labels
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.yaxis.set_visible(True)
        ax.xaxis.set_visible(True)
        plt.ylabel('Frequency')
        plt.tight_layout()
        
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

def create_wordcloud(word_freq, list_name):
    """
    Creates and saves a word cloud image based on the frequency distribution of words
    using an image silhouette as a mask, with increased size for better visibility.

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
        # Here you might need to provide the actual path to your silhouette image
        image_path = next(Path().cwd().joinpath('Images').glob('*.png'))  # Grabbing the first .png image

        # Load the silhouette for the mask
        silhouette_image = Image.open(image_path)
        mask = np.array(silhouette_image)

        # Generate word cloud with increased size
        wordcloud = WordCloud(width=1600, height=800, background_color='white', mask=mask,
                              contour_width=1, contour_color='steelblue').generate_from_frequencies(word_freq)
        
        # Plotting with an increased figure size
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Save and close the plot
        plt.savefig(f'results/{list_name}/wordcloud.png', dpi=300)  # Increased dpi for better resolution
        plt.close()
    except Exception as e:
        logging.error(f"Error in create_wordcloud for list {list_name}: {e}")

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
