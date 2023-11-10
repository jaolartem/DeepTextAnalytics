
import logging
from pathlib import Path
from pdftotext import extract_text_from_pdf, detect_language, get_words_without_stopwords
from Main_analysis import lexical_diversity, pos_tag_frequency, word_network_analysis
from Vizualization import plot_ngrams, create_wordcloud, plot_word_network, plot_lexical_diversity_histogram

# Configure logging to record errors in 'text_anal.log'
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def process_pdf_path(path):
    all_words = []  # List for all words
    words_by_file = {}  # Dictionary for words by file, including language
    words_by_language = {}  # Dictionary for words by language

    def process_pdf_file(pdf_path):
        pdf_text = extract_text_from_pdf(str(pdf_path))
        if pdf_text:
            detected_language = detect_language(pdf_text)
            filtered_words = get_words_without_stopwords(pdf_text, detected_language)

            all_words.extend(filtered_words)
            words_by_file[pdf_path.name] = {
                'language': detected_language,
                'words': filtered_words
            }
            words_by_language.setdefault(detected_language, []).extend(filtered_words)

    path = Path(path)
    if path.is_file() and path.suffix.lower() == '.pdf':
        process_pdf_file(path)
    elif path.is_dir():
        for pdf_file in path.glob('*.pdf'):
            process_pdf_file(pdf_file)

    return words_by_file, all_words, words_by_language

def perform_analysis_and_visualization(words_by_file):
    for file_name, data in words_by_file.items():
        words = data['words']
        # Perform text analysis
        diversity = lexical_diversity(' '.join(words))
        pos_freq = pos_tag_frequency(' '.join(words))
        word_net = word_network_analysis(' '.join(words))

        # Perform visualizations
        plot_ngrams(words, 2, file_name)  # Example: plotting bigrams
        word_freq = Counter(words)
        create_wordcloud(word_freq, file_name)
        plot_word_network(words, file_name)
        plot_lexical_diversity_histogram([diversity], file_name)

def main():
    # Example path
    path = 'path_to_your_pdf_files'
    words_by_file, _, _ = process_pdf_path(path)
    perform_analysis_and_visualization(words_by_file)

if __name__ == "__main__":
    main()
