import logging
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
from pathlib import Path
import uuid

# Ensure that the required NLTK data is available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Configure logging to record errors in 'text_anal.log'



def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    
    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The extracted text from the PDF, or None if an error occurs.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''.join(reader.getPage(page).extractText() for page in range(reader.numPages))
            return text
    except (FileNotFoundError, PyPDF2.utils.PdfReadError) as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return None


def get_language_name_from_code(code):
    """
    Returns the full name of the language in lowercase based on the ISO code.

    Parameters:
        code (str): The ISO language code.

    Returns:
        str: The full name of the language in lowercase, or 'unknown' if not found.
    """
    language_map = {
        'ar': 'arabic', 'az': 'azerbaijani', 'eu': 'basque', 'bn': 'bengali',
        'ca': 'catalan', 'zh': 'chinese', 'da': 'danish', 'nl': 'dutch',
        'en': 'english', 'fi': 'finnish', 'fr': 'french', 'de': 'german',
        'el': 'greek', 'he': 'hebrew', 'hu': 'hungarian', 'id': 'indonesian',
        'it': 'italian', 'kk': 'kazakh', 'ne': 'nepali', 'no': 'norwegian',
        'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian', 'sl': 'slovene',
        'es': 'spanish', 'sv': 'swedish', 'tg': 'tajik', 'tr': 'turkish'
    }
    return language_map.get(code, 'unknown')


def get_words_without_stopwords(text, language_code):
    """
    Removes stopwords from text based on the detected language code.

    Parameters:
        text (str): The text from which to remove stopwords.
        language_code (str): The language code to identify the stopwords.

    Returns:
        list: A list containing the filtered words.
    """
    try:
        language_code = detect(text)
        language = get_language_name_from_code(language_code)            
        stop_words = set(stopwords.words(language))
    except OSError:
        logging.warning(f"Stopwords not found for detected language '{language}'. Using English stopwords as default.")
        stop_words = set(stopwords.words('english'))
    
    words = word_tokenize(text)
    return [word for word in words if word.lower() not in stop_words and word.isalpha()]


def process_pdf_file(pdf_path):
    """
    Processes a single PDF file to extract and filter words.

    Parameters:
        pdf_path (Path): The path object of the PDF file.

    Returns:
        dict: A dictionary with file data including language and filtered words, or None if an error occurs.
    """
    try:
        pdf_text = extract_text_from_pdf(str(pdf_path))
        if pdf_text:
            try:
                language_code = detect(pdf_text)
            except LangDetectException:
                logging.warning(f"Language detection failed for file {pdf_path}")
                language_code = 'unknown'

            filtered_words = get_words_without_stopwords(pdf_text, language_code)
            return {
                'file_name': pdf_path.stem,
                'language': get_language_name_from_code(language_code),
                'words': filtered_words
            }
    except Exception as e:
        logging.error(f"Error processing PDF file {pdf_path}: {e}")
        return None


def process_pdf_path(path):
    """
    Processes a given path to extract words from PDF files. It handles both individual files and directories.

    Parameters:
        path (str): The file path or directory path.

    Returns:
        tuple: A tuple containing the processed data for each file and summary data for all files combined.
    """
    words_by_file = {}
    all_words = []
    language_distribution = {}

    path = Path(path)
    pdf_paths = [path] if path.is_file() else path.glob('*.pdf')

    for pdf_path in pdf_paths:
        file_data = process_pdf_file(pdf_path)
        if file_data:
            # Extend the word list for each language and increment the document count
            language = file_data['language']
            language_distribution.setdefault(language, {'file_name': f"All_records_{language}", 'language': language, 'documents': 0, 'words': []})
            language_distribution[language]['words'].extend(file_data['words'])
            language_distribution[language]['documents'] += 1

            # Add all words to the complete word list
            all_words.extend(file_data['words'])

            # Create a unique entry for each processed file
            unique_id = str(uuid.uuid4())
            words_by_file[unique_id] = file_data

    # Create a summary record for all languages combined
    all_languages = 'all'
    all_records = {
        'file_name': "All_records",
        'language': all_languages,
        'documents': len(words_by_file),
        'words': all_words
    }

    # Add the summary record to the language distribution
    language_distribution[all_languages] = all_records

    return words_by_file, language_distribution


# Example usage and other functions can be placed here.
