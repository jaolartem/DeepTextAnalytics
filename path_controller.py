import logging
from pdftotext import process_pdf_path
from Main_analysis import lexical_diversity, pos_tag_frequency, word_network_analysis
from Vizualization import plot_ngrams, create_wordcloud, plot_word_network, plot_lexical_diversity_histogram
from collections import Counter

# Configure logging to record errors in 'text_anal.log'
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def words(path):
    """
    Process a given path to extract words from PDF files.

    Args:
        path (str): The file path or directory path.

    Returns:
        dict: A dictionary with unique identifiers as keys, and file data as values.
    """
    try:
        words_by_file = process_pdf_path(path)
        return words_by_file
    except Exception as e:
        logging.error(f"Error processing path '{path}': {e}")
        return {}

def perform_analysis_and_visualization(words_by_file):
    """
    Perform text analysis and visualization for each set of words in the dictionary.

    Args:
        words_by_file (dict): Dictionary with unique identifiers and file data.

    Returns:
        None: The function performs analysis and visualization but does not return any value.
    """
    for unique_id, data in words_by_file.items():
        try:
            file_name = data.get('file_name', '')
            words = data.get('words', [])

            if words:
                # Perform text analysis
                diversity = lexical_diversity(' '.join(words))
                pos_freq = pos_tag_frequency(' '.join(words))
                word_net = word_network_analysis(' '.join(words))

                # Perform visualizations
                plot_ngrams(words, 2, file_name)
                word_freq = Counter(words)
                create_wordcloud(word_freq, file_name)
                plot_word_network(words, file_name)
                plot_lexical_diversity_histogram([diversity], file_name)
        except Exception as e:
            logging.error(f"Error in analysis and visualization for file '{file_name}': {e}")

def main():
    """
    Main function to process PDF files, perform analysis, and generate visualizations.
    """
    
    try:
        path = 'DATA'
        words_by_file = words(path)
        perform_analysis_and_visualization(words_by_file)
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()


