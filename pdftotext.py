
import logging
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from pathlib import Path
from collections import Counter
import uuid

# Logging configuration
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
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

# Function to detect the language of a given text
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        logging.error(f"Error detecting language: {e}")
        return "Language detection failed"

# Function to remove stopwords from text in a given language
def get_words_without_stopwords(text, language):
    try:
        stop_words = set(stopwords.words(language))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        return filtered_words
    except LookupError:
        logging.error(f"Stopwords for {language} not found.")
        return []
    except Exception as e:
        logging.error(f"Error filtering stopwords: {e}")
        return []

def process_pdf_path(path):
    words_by_file = {}  # Dictionary for words by file, including language and unique identifier

    def process_pdf_file(pdf_path):
        pdf_text = extract_text_from_pdf(str(pdf_path))
        if pdf_text:
            detected_language = detect_language(pdf_text)
            filtered_words = get_words_without_stopwords(pdf_text, detected_language)

            unique_id = str(uuid.uuid4())
            words_by_file[unique_id] = {
                'file_name': pdf_path.name,
                'language': detected_language,
                'words': filtered_words
            }

    path = Path(path)
    if path.is_file() and path.suffix.lower() == '.pdf':
        process_pdf_file(path)
    elif path.is_dir():
        for pdf_file in path.glob('*.pdf'):
            process_pdf_file(pdf_file)

    return words_by_file

# Other functions and main part of the script would remain the same