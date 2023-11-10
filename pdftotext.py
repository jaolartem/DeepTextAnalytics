import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
import logging

# Logging configuration to record errors in a file named 'text_anal.log'
logging.basicConfig(filename='text_anal.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Download necessary NLTK data (punkt tokenizer models and stopwords)
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            num_pages = reader.numPages
            text = ''
            # Extract text from each page
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
        # Filter words not in stop words and are alphabetic
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        return filtered_words
    except LookupError:
        logging.error(f"Stopwords for {language} not found.")
        return []
    except Exception as e:
        logging.error(f"Error filtering stopwords: {e}")
        return []

# Path to the PDF file
pdf_path = 'your_file.pdf'

# Extract text and filter words
pdf_text = extract_text_from_pdf(pdf_path)
if pdf_text:
    detected_language = detect_language(pdf_text)
    filtered_words = get_words_without_stopwords(pdf_text, detected_language)
    print(filtered_words)
else:
    print("Error extracting text from PDF.")
