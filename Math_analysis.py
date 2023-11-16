import logging
from nltk import word_tokenize, pos_tag
from nltk.probability import FreqDist
from pdftotext import get_language_name_from_code
import pyphen
from langdetect import detect
from collections import Counter
import networkx as nx
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from itertools import combinations
from nltk.tokenize import sent_tokenize




def analyze_collocations(words):
    """
    Identifies common bigram collocations in the list of words.

    Args:
        words (list): List of tokenized words.

    Returns:
        list: List of bigram collocations.
    """
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    return finder.nbest(bigram_measures.pmi, 10)  # Top 10 collocations


def lexical_diversity(text):
    """
    Calculates the lexical diversity of the given text.
    Lexical diversity is the ratio of unique words to the total number of words,
    considering the case-insensitive uniqueness of words.

    Parameters:
    text (str): The text to analyze.

    Returns:
    float: The lexical diversity of the text. Returns 0 if the text is empty or not a string.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")

        words = word_tokenize(text.lower())  # Tokenize and convert to lower case
        unique_words = set(words)
        return len(unique_words) / len(words)
    except Exception as e:
        logging.error(f"Error in lexical_diversity: {e}")
        return 0

def pos_tag_frequency(text):
    """
    Calculates the frequency distribution of part-of-speech tags in the given text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    Counter: A counter object of part-of-speech tags.
             Returns an empty Counter if the text is empty or an error occurs.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")

        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        pos_tag_freq = Counter(tag for (word, tag) in pos_tags)
        return pos_tag_freq
    except Exception as e:
        logging.error(f"Error in pos_tag_frequency: {e}")
        return Counter()

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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")

        words = word_tokenize(text)
        G = nx.Graph()
        G.add_edges_from(ngrams(words, 2))
        return G
    except Exception as e:
        logging.error(f"Error in word_network_analysis: {e}")
        return nx.Graph()
    
def ngram_analysis(text, n):
    """
    Analyzes the frequency of n-grams in the given text, from 1-gram to n-gram.

    Parameters:
    text (str): The text to analyze.
    n (int): The maximum size of n-gram.

    Returns:
    dict: A dictionary where keys are the n-gram size and values are Counter objects
          with the most common n-grams and their frequencies.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")

        words = word_tokenize(text.lower())  # Tokenize and convert to lower case
        ngram_freq = {}

        for i in range(1, n + 1):
            n_grams = ngrams(words, i)
            ngram_freq[i] = Counter(n_grams).most_common()

        return ngram_freq
    except Exception as e:
        logging.error(f"Error in ngram_analysis for n={n}: {e}")
        return {}

def pos_tagging(words):
    """
    Performs Part-of-Speech tagging on the list of words.
    
    Args:
        words (list): List of tokenized words.

    Returns:
        list: List of tuples with word and its POS tag.
    """
    return pos_tag(words)

def analyze_collocations(words):
    """
    Identifies common bigram collocations in the list of words.

    Args:
        words (list): List of tokenized words.

    Returns:
        list: List of bigram collocations.
    """
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    return finder.nbest(bigram_measures.pmi, 10)  # Top 10 collocations

def word_length_analysis(words):
    """
    Analyzes the length of words in the list.

    Args:
        words (list): List of tokenized words.

    Returns:
        dict: A dictionary with word lengths and their frequencies.
    """
    length_distribution = {}
    for word in words:
        length = len(word)
        length_distribution[length] = length_distribution.get(length, 0) + 1
    return length_distribution
def co_occurrence_analysis(words, window_size=2):
    """
    Analyzes co-occurrence of words within a specified window size.

    Args:
        words (list): List of tokenized words.
        window_size (int): Size of the window to consider for co-occurrence.

    Returns:
        dict: A dictionary of co-occurring word pairs and their frequencies.
    """
    co_occurrence = {}
    for i in range(len(words) - window_size + 1):
        for pair in combinations(words[i:i + window_size], 2):
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
    return co_occurrence



def count_syllables(word, language_code):
    """
    Counts the number of syllables in a given word using Pyphen library.

    Args:
        word (str): The word to count syllables in.
        language_code (str): ISO 639-1/2 language code.

    Returns:
        int: Number of syllables in the word.
    """
    try:
        dic = pyphen.Pyphen(lang=language_code)
        hyphenated = dic.inserted(word)
        return len(hyphenated.split('-'))
    except Exception as e:
        logging.error(f"Error in count_syllables: {e}")
        return 0  # Devuelve 0 en caso de error


def readability_index(text):
    """
    Calculates the Flesch Reading Ease score for the given text.

    Args:
        text (str): Text to analyze.

    Returns:
        float: Flesch Reading Ease score.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return 0  # Devuelve 0 para texto vacío o no válido

        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        if not sentences or not words:
            return 0  # Evita división por cero

        language_code = detect(text)
        syllable_count = sum([count_syllables(word, language_code) for word in words])
        words_per_sentence = len(words) / len(sentences)
        return 206.835 - 1.015 * words_per_sentence - 84.6 * (syllable_count / len(words))
    except Exception as e:
        logging.error(f"Error in readability_index: {e}")
        return 0


