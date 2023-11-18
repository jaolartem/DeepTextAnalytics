import logging
import pandas as pd

from pathlib import Path
from pdftotext import process_pdf_path
from Math_analysis import lexical_diversity, pos_tag_frequency, word_network_analysis, analyze_collocations,readability_index  
from Vizualization import plot_ngrams, create_wordcloud, plot_word_network, plot_lexical_diversity_histogram, plot_pos_frequency_distribution, plot_word_length_distribution, plot_readability_index, create_wordcloud_multi
from collections import Counter

logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def analyze_single_document(data):
    """
    Analyzes a single document by computing various textual metrics and generating visualizations.

    Args:
        data (dict): A dictionary containing the file name and a list of words from the document.

    Returns:
        dict: The updated data dictionary with added analysis results.

    Raises:
        ValueError: If 'words' in data is not a list or is empty.
    """
    try:
        file_name = data.get('file_name', '')
        words = data.get('words', [])

        # Validate that words is a list and is not empty
        if not isinstance(words, list) or not words:
            raise ValueError(f"'words' must be a non-empty list. Found: {type(words)}")

        # Join words into a single string for certain analyses
        text = ' '.join(words)
        
        # Perform various textual analyses
        data['diversity'] = lexical_diversity(text)
        data['pos_freq'] = pos_tag_frequency(text)
        data['word_net'] = word_network_analysis(text)
        data['collocations'] = analyze_collocations(words)
        data['readability_index'] = readability_index(text)

        # Generate visualizations
        numbers = [1, 2, 3, 4, 5]
        
        for n in numbers:
            plot_ngrams(words, n, file_name )
            
        word_freq = Counter(words)
        create_wordcloud(word_freq, file_name)
        plot_word_network(words, file_name)

    except Exception as e:
        error_message = f"Error in analyze_single_document for file '{file_name}': {e}"
        logging.error(error_message)
        # You might want to add more sophisticated error handling here
        data['error'] = error_message

    return data



def analyze_document_set(words_by_file):
    """
    Performs text analysis for each document in the set, aggregates analysis results, and saves them as a CSV file.

    Args:
        words_by_file (dict): Dictionary with unique identifiers and file data.

    Returns:
        dict: Dictionary with updated analysis for each document.
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
        all_pos_frequencies.update(updated_data.get('pos_freq', Counter()))
        all_word_lengths.extend([len(word) for word in updated_data.get('words', [])])
        all_readability_scores.append(updated_data.get('readability_index', 0))

  
    # Convert to DataFrame for individual documents
    df = pd.DataFrame.from_dict(words_by_file, orient='index')

    # Filter out columns with complex data structures like lists, tuples, and dictionaries
    scalar_columns = [col for col in df.columns if not isinstance(df[col].iloc[0], (list, tuple, dict))]
    df_filtered = df[scalar_columns]
       # Save as CSV
    output_path = 'results/document_analysis.csv'
    df_filtered.to_csv(output_path, index=False)


    # Perform and visualize aggregated analysis
    plot_lexical_diversity_histogram(all_diversities, 'aggregated')
    plot_pos_frequency_distribution(all_pos_frequencies, 'aggregated')
    plot_word_length_distribution(Counter(all_word_lengths), 'aggregated')
    plot_readability_index(all_readability_scores, 'aggregated')

    return words_by_file
  
def analyze_language_distribution(language_distribution):
    """
    Perform basic analysis and visualization for each language in the distribution.

    Args:
        language_distribution (dict): Dictionary with language as keys and word lists as values.

    Raises:
        ValueError: If the word list for any language is not a non-empty list.
    """
    # Check that language_distribution is a dictionary
    if not isinstance(language_distribution, dict):
        raise ValueError("language_distribution must be a dictionary.")

    # Iterate over each language in the distribution
    for language, data in language_distribution.items():
        try:    
            
            words = data.get('words', [])
   
            # Validate that words is a list and is not empty
            if not isinstance(words, list) or not words:
                raise ValueError(f"'words' must be a non-empty list. Found: {type(words)} for language '{language}'")
            
            numbers = [1, 2, 3, 4, 5]        
            for n in numbers:
                plot_ngrams(words, n, language) 

            # Generate frequency distribution of words
            word_freq = Counter(words)

            # Perform visualizations for the language
            create_wordcloud_multi(word_freq, language)
            plot_word_network(words, language)

        except Exception as e:
            # Log any errors that occur during the analysis and visualization
            logging.error(f"Error in analysis and visualization for language '{language}': {e}")


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
        logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')
        
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
    
