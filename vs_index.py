"""
boolean_index.py
author: Zhiheng Wang

Build a inverted index with weight from a json file of Wiki 2018 movies and save it as shelve.
"""
import json
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import shelve
import math


# input json file
# output a dataset in dictionary type.
def load_json_file_to_dict(file_path):
    with open(file_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
    return data


# tokenize text and remove stop words in dictionary
# input string text
# output list words
def tokenize(text):
    # make a stop word list including NLTK stop words and punctuation
    stop = stopwords.words('english') + list(string.punctuation)

    # tokenize text without stopwords or punctuation
    words = [token for token in word_tokenize(text.lower()) if token not in stop]

    return words


# apply PorterStemm on word
# input string word
# output stemmed word
def stemming(word):
    porter = PorterStemmer()
    return porter.stem(word)


def build_inverted_index(file_path):
    """
    input file path
    output (inverted index, idf, tfidf) and normalized_length
    """
    # load json data
    data = load_json_file_to_dict(file_path)
    inverted_index = {}
    inverted_index_tfidf = {}
    normalized_length = {}
    N = float(len(data.keys()))

    # if list is not empty, convert list to string
    for film in range(1, len(data.keys())):

        text = data[str(film)]["Text"]
        if isinstance(text, list):
            text = ' '.join(text)
        tittle = data[str(film)]["Title"]
        if isinstance(tittle, list):
            tittle = ' '.join(tittle)

        # Index and query over title and text only
        words = tokenize(text) + tokenize(tittle)

        # Iterate list to do stemming on each word
        for i, word in enumerate(words):
            words[i] = stemming(word)

        for i, word in enumerate(words):

            if inverted_index.get(words[i], None) is None:
                # count term frequency in a single document in (film, tf)
                inverted_index[words[i]] = {(str(film), words.count(word))}

            else:
                inverted_index[words[i]].add((str(film), words.count(word)))

    # print(inverted_index)

    # # iterate inverted index to get document freq for each vocab
    # for key, value in inverted_index.items():
    #     document_freq[key] = len(value)

    # calculate idf and tf-idf and save it as tuple
    for k, v in inverted_index.items():

        idf = math.log10((N/float(len(v)))) if len(v) != 0.0 else 0.0

        # for every (film, tf), calculate its tf_idf
        for (film, tf) in v:
            w_tf = ( 1.0 + math.log10(tf) ) if tf != 0.0 else 0.0
            tf_idf = w_tf * idf

            # create a new dict to store (film, idf, tf-idf)
            # when we compute the query's weight, we need to resue idf but tf becomes useless
            # so we only save idf and tf-idf to our inverted index
            if inverted_index_tfidf.get(k, None) is None:
                inverted_index_tfidf[k] = {(film, idf, tf_idf)}
            else:
                inverted_index_tfidf[k].add((film, idf, tf_idf))

    # print(inverted_index_tfidf)

    # iterate inverted index tfidf to get sum of squared length for every document
    for vocab, posting_list in inverted_index_tfidf.items():
        for (film, idf, tf_idf) in posting_list:
            if normalized_length.get(film, None) is None:
                normalized_length[film] = tf_idf * tf_idf
            else:
                tf_idf_square = tf_idf * tf_idf
                normalized_length[film] += tf_idf_square

    # take the square root to finish the normalization for length
    for id, sum_of_squared_tf_idf in normalized_length.items():
        normalized_length[id] = math.sqrt(sum_of_squared_tf_idf)

    return inverted_index_tfidf, normalized_length


# save inverted indices and normalized length to shelve
def save_to_shelve(file_path):
    inverted_index, normalized_length = build_inverted_index(file_path)
    a = shelve.open('inverted_index', flag='c', protocol=None, writeback=False)
    for key, value in inverted_index.items():
        a[key] = value
    a.close()

    b = shelve.open('normalized_length', flag='c', protocol=None, writeback=False)
    for key, value in normalized_length.items():
        b[key] = value
    b.close()


def save_2018_movie_json_to_shelve(file_path):
    data = load_json_file_to_dict(file_path)
    db = shelve.open('2018_movies_database', flag='c', protocol=None, writeback=False)
    for key, value in data.items():
        db[key] = value
    db.close()


# file_path = 'test_corpus.json'
file_path = '2018_movies.json'
save_to_shelve(file_path)
save_2018_movie_json_to_shelve(file_path)

