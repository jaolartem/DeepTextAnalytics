import logging
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from pathlib import Path
from collections import Counter
import uuid

nltk.download('stopwords')


# Configure logging to record errors in 'text_anal.log'
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            num_pages = reader.numPages
            text = ''
            for page in range(num_pages):
                text += reader.getPage(page).extractText()
            return text
    except FileNotFoundError:
        logging.error(f"PDF file not found: {pdf_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return None

def get_words_without_stopwords(text):
    """
    Remove stopwords from text and detect its language.

    Args:
        text (str): Text from which to remove stopwords.

    Returns:
        tuple: A tuple containing the filtered words and the detected language.
    """
    try:        
        language = detect(text)
        try:
            stop_words = set(stopwords.words(language))
        except LookupError:
            logging.warning(f"Stopwords not found for detected language '{language}'. Using English stopwords as default.")
            stop_words = set(stopwords.words('english'))
        
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        return filtered_words, language
    except Exception as e:
        logging.error(f"Error in get_words_without_stopwords: {e}")
        return [], None

    except LookupError:
        logging.error(f"Stopwords for detected language not found.")
        return [], None
    except Exception as e:
        logging.error(f"Error in get_words_without_stopwords: {e}")
        return [], None

def process_pdf_path(path):
    """
    Process a given path to extract words from PDF files.

    Args:
        path (str): The file path or directory path.

    Returns:
        dict: A dictionary with unique identifiers as keys, and file data as values.
    """
    words_by_file = {}

    def process_pdf_file(pdf_path):
        try:
            pdf_text = extract_text_from_pdf(str(pdf_path))
            if pdf_text:
                filtered_words, language = get_words_without_stopwords(pdf_text)

                unique_id = str(uuid.uuid4())
                words_by_file[unique_id] = {
                    'file_name': pdf_path.stem,
                    'language': language,
                    'words': filtered_words
                }
        except Exception as e:
            logging.error(f"Error processing PDF file {pdf_path}: {e}")

    try:
        path = Path(path)
        if path.is_file() and path.suffix.lower() == '.pdf':
            process_pdf_file(path)
        elif path.is_dir():
            for pdf_file in path.glob('*.pdf'):
                process_pdf_file(pdf_file)
    except Exception as e:
        logging.error(f"Error processing path {path}: {e}")

    return words_by_file

# Main function and perform_analysis_and_visualization remain the same
