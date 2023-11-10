import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud

# Necesario para el análisis de categorías gramaticales
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Función para plotear gráficos de barras para n-gramas
def plot_ngrams(words, n, list_name):
    n_grams = list(ngrams(words, n))
    ngram_freq = Counter(n_grams)
    top_ngrams = ngram_freq.most_common(10)

    labels, values = zip(*top_ngrams)
    indexes = range(len(labels))

    plt.figure(figsize=(10, 8))
    plt.bar(indexes, values, align='center')
    plt.xticks(indexes, [' '.join(label) for label in labels], rotation='vertical')
    plt.tight_layout()

    plt.savefig(f'results/{list_name}/top_{n}_grams.png')
    plt.close()

# Función para crear una nube de palabras
def create_wordcloud(word_freq, list_name):
    images = os.listdir('Images')
    image_path = os.path.join('Images', random.choice(images))

    wordcloud = WordCloud(width=800, height=800, background_color='white', mask=image_path).generate_from_frequencies(word_freq)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(f'results/{list_name}/wordcloud.png')
    plt.close()

# Función para plotear un gráfico de red de palabras
def plot_word_network(words, list_name):
    G = nx.Graph()
    G.add_edges_from(ngrams(words, 2))

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray')

    plt.savefig(f'results/{list_name}/word_network.png')
    plt.close()

# Función para plotear un histograma de diversidad léxica
def plot_lexical_diversity_histogram(diversities, list_name):
    plt.figure(figsize=(10, 6))
    plt.hist(diversities, bins=20, color='skyblue')
    plt.title('Histograma de Diversidad Léxica')
    plt.xlabel('Diversidad Léxica')
    plt.ylabel('Frecuencia')

    plt.savefig(f'results/{list_name}/lexical_diversity_histogram.png')
    plt.close()

# Función para integrar y guardar todos los análisis
def integrate_analysis_and_save(words_dict, list_name):
    os.makedirs(f'results/{list_name}', exist_ok=True)

    for n in range(1, 6):  # De 1-grama a 5-grama
        plot_ngrams(words_dict[list_name], n, list_name)

    # Crear y guardar la nube de palabras
    word_freq = Counter(words_dict[list_name])
    create_wordcloud(word_freq, list_name)

    # Crear y guardar el gráfico de red
    plot_word_network(words_dict[list_name], list_name)

    # Calcular y guardar el histograma de diversidad léxica
    diversity = [len(set(words)) / len(words) if words else 0 for words in words_dict.values()]
    plot_lexical_diversity_histogram(diversity, list_name)

# Ejemplo de uso
# words_dict = {'my_list': ['your', 'sample', 'text', 'goes', 'here']}
# integrate_analysis_and_save(words_dict, 'my_list')
