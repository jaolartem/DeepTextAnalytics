import logging
from pathlib import Path
from pdftotext import extract_text_from_pdf, detect_language, get_words_without_stopwords

# Configure logging to record errors in 'text_anal.log'
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def process_pdf_path(path):
    """
    Process a given path to extract words from PDF files.
    
    If the path is a single PDF file, processes this file.
    If the path is a directory, processes all PDF files within it.
    
    Args:
    path (str): The file path or directory path.
    
    Returns:
    tuple: Containing three elements:
        1. Dictionary with words by file including the detected language.
        2. List of all words extracted from all PDFs.
        3. Dictionary of words categorized by the detected language.
    """

    all_words = []  # List for all words
    words_by_file = {}  # Dictionary for words by file, including language
    words_by_language = {}  # Dictionary for words by language

    def process_pdf_file(pdf_path):
        """Helper function to process a single PDF file."""
        try:
            pdf_text = extract_text_from_pdf(str(pdf_path))
            if pdf_text:
                detected_language = detect_language(pdf_text)
                filtered_words = get_words_without_stopwords(pdf_text, detected_language)

                # Update the word containers
                all_words.extend(filtered_words)
                words_by_file[pdf_path.name] = {
                    'language': detected_language,
                    'words': filtered_words
                }
                words_by_language.setdefault(detected_language, []).extend(filtered_words)
        except Exception as e:
            logging.error(f"Error processing PDF file {pdf_path}: {e}")

    try:
        path = Path(path)

        if path.is_file() and path.suffix.lower() == '.pdf':
            process_pdf_file(path)
        elif path.is_dir():
            for pdf_file in path.glob('*.pdf'):
                process_pdf_file(pdf_file)
        else:
            logging.error(f"The path provided is neither a PDF file nor a directory: {path}")
    except Exception as e:
        logging.error(f"Error processing the path {path}: {e}")

    return words_by_file, all_words, words_by_language

# Example of how to use the function (replace 'your_path' with the actual file or folder path):
# results = process_pdf_path('your_path')
# print(results)
