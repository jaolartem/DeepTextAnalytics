import logging
import pandas as pd

from pathlib import Path
from pdftotext import process_pdf_path
from Math_analysis import lexical_diversity, pos_tag_frequency, word_network_analysis, analyze_collocations,readability_index  
from Vizualization import plot_ngrams, create_wordcloud, plot_word_network, plot_lexical_diversity_histogram, plot_pos_frequency_distribution, plot_word_length_distribution, plot_readability_index
from collections import Counter

logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def analyze_single_document(data):
    try:
        file_name = data.get('file_name', '')
        words = data.get('words', [])

        if words:
            text = ' '.join(words)  # Convertir la lista de palabras en una cadena de texto
            data['diversity'] = lexical_diversity(text)
            data['pos_freq'] = pos_tag_frequency(text)
            data['word_net'] = word_network_analysis(text)
            data['collocations'] = analyze_collocations(words)
            data['relea_index'] = readability_index(text)  # Pasar el texto completo

            plot_ngrams(words, 5, file_name)
            word_freq = Counter(words)
            create_wordcloud(word_freq, file_name)
            plot_word_network(words, file_name)
    except Exception as e:
        logging.error(f"Error in analyze_single_document for file '{file_name}': {e}")
    return data


def analyze_document_set(words_by_file):
    """
    Performs text analysis for each document in the set, aggregates analysis results, and saves them as a CSV file.

    Args:
        words_by_file (dict): Dictionary with unique identifiers and file data.

    Returns:
        None: This function performs analysis and saves results but does not return any value.
    """
    # Variables to store aggregated metrics
    all_diversities = []
    all_pos_frequencies = Counter()
    all_word_lengths = []
    all_readability_scores = []

    for unique_id, data in words_by_file.items():
        updated_data = analyze_single_document(data)
        words_by_file[unique_id] = updated_data

        # Add metrics to the aggregate lists
        all_diversities.append(updated_data.get('diversity', 0))
        all_pos_frequencies.update(updated_data.get('pos_freq', {}))
        all_word_lengths.extend([len(word) for word in updated_data.get('words', [])])
        all_readability_scores.append(updated_data.get('readability_index', 0))

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(words_by_file, orient='index')
    df_filtered = df[['diversity', 'pos_freq', 'word_net']]  # Adjust based on available columns

    # Save as CSV
    output_path = Path('results/document_analysis.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)

    # Perform and visualize aggregated analysis
    plot_lexical_diversity_histogram(all_diversities, 'aggregated')
    plot_pos_frequency_distribution(all_pos_frequencies, 'aggregated')
    plot_word_length_distribution(Counter(all_word_lengths), 'aggregated')
    plot_readability_index(all_readability_scores, 'aggregated')

    return words_by_file
  

    return words_by_file

# La función 'plot_word_length_distribution' debe estar definida en Visualization.py

def analyze_language_distribution(language_distribution):
    """
    Perform basic analysis and visualization for each language in the distribution.

    Args:
        language_distribution (dict): Dictionary with language as keys and word lists as values.

    Returns:
        None: The function performs analysis and visualization but does not return any value.
    """
    for language, words in language_distribution.items():
        try:
            # Basic visualizations for each language
            word_freq = Counter(words)
            create_wordcloud(word_freq, language)
            plot_word_network(words, language)
        except Exception as e:
            logging.error(f"Error in basic analysis and visualization for language '{language}': {e}")

def words(path):
    """
    Process a given path to extract words from PDF files.

    Args:
        path (str): The file path or directory path.

    Returns:
        dict: A dictionary with unique identifiers as keys, and file data as values.
    """
    try:
        words_by_file, language_distribution = process_pdf_path(path)
        return words_by_file, language_distribution
    except Exception as e:
        logging.error(f"Error processing path '{path}': {e}")
        return {}

def main(path):
    try:
        words_by_file, language_distribution = words(path)

        # Perform detailed analysis on each document
        words_by_file = analyze_document_set(words_by_file)

        # Perform basic analysis on language distribution
        analyze_language_distribution(language_distribution)

    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    path = "DATA"  
    main(path)
    
