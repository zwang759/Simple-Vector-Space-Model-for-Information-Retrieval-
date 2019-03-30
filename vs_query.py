from flask import Flask, render_template, request
from nltk.corpus import stopwords
import string
from vs_index import stemming
import shelve
import nltk
import heapq
import math
# from itertools import groupby
from nltk import word_tokenize


def tokenize(text):
    # make a stop word list including NLTK stop words and punctuation
    stop = stopwords.words('english') + list(string.punctuation)

    # tokenize text without stopwords or punctuation
    words = [token for token in word_tokenize(text.lower()) if token not in stop]

    return words


def search(query, unknown_term_list, query_terms_occured):
    """
    Input query
    Return a list of (cosine_score, movie_ids) that
    match the query ranked by cosine_score.
    """
    cosine_score = []
    qi_multiply_di = []
    inverted_index = shelve.open('inverted_index')
    normalized_length = shelve.open('normalized_length')
    tokens = tokenize(query)

    # add occured query tokens to a list and normalize query tokens
    for i, token in enumerate(tokens):
        if stemming(token) in inverted_index.keys():
            if token not in query_terms_occured:
                query_terms_occured.append(token)

        else:
            if token not in unknown_term_list:
                unknown_term_list.append(token)

        tokens[i] = stemming(token)

    # if no query terms occur, return empty list
    if not query_terms_occured:
        return list(), list()

    for token in tokens:
        query_tf = 1.0 + math.log10(tokens.count(token))
        if token in inverted_index.keys():
            for (doc_id, d_idf, d_tfidf) in inverted_index[token]:
                query_tfidf = query_tf * d_idf
                # append all matched doc_id for all query terms in the form of [doc_id, qi*di] to the list
                qi_multiply_di.append([ doc_id, query_tfidf * d_tfidf ])

    # # group by doc_id and sum their qi*di, then divided by the normalized length of this doc
    # for i, g in groupby(sorted(qi_multiply_di), key=lambda x: x[0]):
    #     # change the format to [qi*di, doc_id]
    #     cosine_score.append([ sum(v[1] for v in g) / normalized_length[i], i ])

    doc_score = {}
    for doc_id, score in qi_multiply_di:
        if doc_id not in doc_score.keys():
            doc_score[doc_id] = score
        else:
            doc_score[doc_id] += score
    # print('doc_score dict is ')
    # print(doc_score)
    for doc_id, score in doc_score.items():
        # change the format to [qi*di, doc_id]
        # print('length is ')
        # print(normalized_length[doc_id])
        cosine_score.append([score/normalized_length[doc_id], doc_id])

    # print('cosine score is ')
    # print(cosine_score)
    heapq.heapify(cosine_score)
    sorted(unknown_term_list)
    return heapq.nlargest(len(cosine_score), cosine_score)


def dummy_movie_data(doc_id):
    """
    Return data fields for a movie.
    Your code should use the doc_id as the key to access the shelf entry for the movie doc_data.
    You can decide which fields to display, but include at least title and text.
    """
    shelf = shelve.open("2018_movies_database", flag='r')
    movie_object = {"title": shelf[doc_id]['Title'],
                    "director": shelf[doc_id]['Director'],
                    "starring": shelf[doc_id]['Starring'],
                    "location": shelf[doc_id]['Location'],
                    "country": shelf[doc_id]['Country'],
                    "running_time": shelf[doc_id]['Running Time'],
                    "text": shelf[doc_id]['Text']
                    }
    return movie_object


def dummy_movie_snippet(doc_id, cosine_score):
    """
    Return a snippet for the results page
    including doc_id, cosine score, title, and a short description (first 3 sentences).
    """
    movie = dummy_movie_data(doc_id)
    sentences = nltk.sent_tokenize(movie['text'])
    description = ''
    if len(sentences) < 1:
        description = 'Missing movie description'
    else:
        if len(sentences) >= 1:
            description= description + sentences[0]
            if len(sentences) >= 2:
                description = description + " " + sentences[1]
                if len(sentences) >= 3:
                    description = description + " " + sentences[2]
    return doc_id, cosine_score, ''.join(movie['title']), description


# Create an instance of the flask application within the appropriate namespace (__name__).
# By default, the application will be listening for requests on port 5000 and assuming the base
# directory for the resource is the directory where this module resides.
app = Flask(__name__)


# Welcome page
# Python "decorators" are used by flask to associate url routes to functions.
# A route is the path from the base directory (as it would appear in the url)
# This decorator ties the top level url "localhost:5000" to the query function, which
# renders the query_page.html template.
@app.route("/")
def query():
    """For top level route ("/"), simply present a query page."""

    return render_template('query_page.html')


# This takes the form data produced by submitting a query page request and returns a page displaying
# results (SERP).
@app.route("/results", methods=['POST'])
def results():
    """Generate a result set for a query and present the 10 results starting with <page_num>."""

    page_num = int(request.form['page_num'])
    query = request.form['query']  # Get the raw user query

    # Get a ranked list of (score, doc_ids) that satisfy the query, and a unknown_terms_list
    unknown_terms_list = []
    query_terms_occured = []
    movie_ids_and_scores = search(query, unknown_terms_list, query_terms_occured)

    stop = stopwords.words('english') + list(string.punctuation)
    skipped = [token for token in word_tokenize(query) if token in stop]
    print(skipped)

    # check if any unknow_terms found during search
    if not query_terms_occured:
        return render_template('error_page.html', unknown_terms=unknown_terms_list, skipped_words=skipped)

    # render the results page
    num_hits = len(movie_ids_and_scores)  # Save the number of hits to display later
    movie_ids_and_scores = movie_ids_and_scores[((page_num - 1) * 10):(page_num * 10)]  # Limit of 10 results per page
    # movie_results = list(map(dummy_movie_snippet, movie_ids))  # Get movie snippets: title, abstract, etc.
    # Using list comprehension:
    movie_results = [dummy_movie_snippet(doc_id, score) for [score, doc_id] in movie_ids_and_scores]
    return render_template('results_page.html', orig_query=query, movie_results=movie_results, srpn=page_num,
                           len=len(movie_ids_and_scores), skipped_words=skipped, query_terms_occured=query_terms_occured,
                           unknown_terms=unknown_terms_list, total_hits=num_hits)


# Process requests for movie_data pages
# This decorator uses a parameter in the url to indicate the doc_id of the film to be displayed
@app.route('/movie_data/<film_id>')
def movie_data(film_id):
    """Given the doc_id for a film, present the title and text (optionally structured fields as well)
    for the movie."""
    data = dummy_movie_data(str(film_id))  # Get all of the info for a single movie
    movie_as_query = ''.join(data['title']) + ' ' + data['text']
    return render_template('doc_data_page.html', data=data, movie_as_query=movie_as_query )


# If this module is called in the main namespace, invoke app.run().
# This starts the local web service that will be listening for requests on port 5000.
if __name__ == "__main__":
    app.run(debug=False)
    # While you are debugging, set app.debug to True, so that the server app will reload
    # the code whenever you make a change.  Set parameter to false (default) when you are
    # done debugging.
