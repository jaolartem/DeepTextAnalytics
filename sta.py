import os
import nltk
import networkx as nx
from collections import Counter
from nltk.util import ngrams

# Necesario para el análisis de categorías gramaticales
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def lexical_diversity(text, list_name):
    """Calculate and save the lexical diversity of a text."""
    words = text.split()
    unique_words = set(words)
    diversity = len(unique_words) / len(words) if words else 0
    with open(f'results/{list_name}/lexical_diversity.txt', 'w') as file:
        file.write(f"Diversidad Léxica: {diversity}\n")
    return diversity

def pos_tag_frequency(text, list_name):
    """Analyze and save the frequency of parts of speech in a text."""
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    tag_freq = nltk.FreqDist(tag for (word, tag) in pos_tags)
    with open(f'results/{list_name}/pos_tag_frequency.txt', 'w') as file:
        for tag, freq in tag_freq.items():
            file.write(f"{tag}: {freq}\n")
    return tag_freq


def integrate_analysis_and_save(text, list_name):
    """Integrate various text analyses and save the results to individual files."""
    os.makedirs(f'results/{list_name}', exist_ok=True)
    
    # Perform and save analyses
    lexical_diversity(text, list_name)
    pos_tag_frequency(text, list_name)
    word_network_analysis(text, list_name)

# Example usage
# text = "Your sample text goes here."
# integrate_analysis_and_save(text, 'my_text_analysis')