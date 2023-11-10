import logging
from nltk import word_tokenize, pos_tag, FreqDist
from collections import Counter
import networkx as nx
from nltk.util import ngrams

# Configure logging to record errors
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def lexical_diversity(text):
    """
    Calculates the lexical diversity of the given text.

    Lexical diversity is defined as the ratio of unique words to the total number of words.

    Parameters:
    text (str): The text to analyze.

    Returns:
    float: The lexical diversity of the text. Returns 0 if the text is empty or not a string.
    """
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        words = text.split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0
    except Exception as e:
        logging.error(f"Error in lexical_diversity: {e}")
        return 0

def pos_tag_frequency(text):
    """
    Calculates the frequency distribution of part-of-speech tags in the given text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    FreqDist: A frequency distribution of part-of-speech tags.
              Returns an empty FreqDist if the text is empty or an error occurs.
    """
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        return FreqDist(tag for (word, tag) in pos_tags)
    except Exception as e:
        logging.error(f"Error in pos_tag_frequency: {e}")
        return FreqDist()

def word_network_analysis(text):
    """
    Creates a network graph based on the bigrams of the words in the given text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    networkx.Graph: A graph object representing word connections. 
                    Returns an empty graph if the text is empty or an error occurs.
    """
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        words = word_tokenize(text)
        G = nx.Graph()
        G.add_edges_from(ngrams(words, 2))
        return G
    except Exception as e:
        logging.error(f"Error in word_network_analysis: {e}")
        return nx.Graph()
