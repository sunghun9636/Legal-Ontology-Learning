from collections import defaultdict
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


def remove_low_frequent_words(documents, percentage):
    # Checks if documents is a list of lists of strings
    assert type(documents) == list
    assert all(map(lambda x: type(x) == list, documents))
    assert all(map(lambda document: all(map(lambda word: type(word) == str, document)), documents))

    frequency_map = defaultdict(int)
    for text in documents:
        word_set = set()
        for token in text:
            word_set.add(token)
        for token in word_set:
            frequency_map[token] += 1

    # include words that are in more than 25% of whole documents
    filtered_documents = \
        [[token for token in text if frequency_map[token] > len(documents)*percentage] for text in documents]
    return filtered_documents


def get_tfidf(documents):
    # Checks if documents is a list of lists of strings
    assert type(documents) == list
    assert all(map(lambda x: type(x) == list, documents))
    assert all(map(lambda document: all(map(lambda word: type(word) == str, document)), documents))

    dictionary = Dictionary(documents)
    # n_items = len(dictionary) # number of total words
    corpus = [dictionary.doc2bow(text) for text in documents]
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # ds = []
    # for doc in corpus_tfidf:
    #     d = [0] * n_items
    #     for index, value in doc:
    #         d[index] = value
    #     ds.append(d)
    # return ds
    return {
        "corpus_tfidf": corpus_tfidf,
        "index2word": dictionary
    }
