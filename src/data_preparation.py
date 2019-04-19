import pickle
from collections import defaultdict
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


def remove_low_high_frequent_words(documents, lower_limit, upper_limit):
    # Checks if documents is a list of lists of strings
    assert type(documents) == list
    assert all(map(lambda x: type(x) == list, documents))
    assert all(map(lambda document: all(map(lambda word: type(word) == str, document)), documents))

    frequency_map = defaultdict(int)
    for text in documents:
        word_set = set()
        for token in text:  # in order to count A WORD once only per document
            word_set.add(token)
        for token in word_set:
            frequency_map[token] += 1

    # e.g include words that are in more than 25% of whole documents
    filtered_documents = \
        [[token for token in text if len(documents) * lower_limit < frequency_map[token] < len(documents) * upper_limit]
         for text in documents]
    return filtered_documents


def take_second(elem):
    return elem[1]


# According to threshold and using TF-IDF of the terms, get the top keywords of the document
def get_important_words(document, threshold):
    # document is list of (index, tf-idf)
    document.sort(key=take_second, reverse=True)  # reverse sort the document w.r.t tf-idf (reverse order)

    important_words = []
    for i in range(int(len(document) * threshold)):
        important_words.append(document[i][0])

    return important_words  # returning the list of important words' index (of the given document)


# According to the threshold and using TF-IDF of the terms, remove non-important terms from each & return new documents
def extract_important_words_tfidf(documents, threshold):
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]

    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    new_documents = []

    for i, doc in enumerate(documents):
        important_words_index = get_important_words(corpus_tfidf[i], threshold)
        new_doc = []
        for term in doc:
            if dictionary.doc2idx([term])[0] in important_words_index:
                new_doc.append(term)
        new_documents.append(new_doc)

    return new_documents  # list of list of terms


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


def main():
    with open('data/case_documents_20000.data', 'rb') as file:
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    print(extract_important_words_tfidf(documents, 0.60)[0])
    print(documents[0])


if __name__ == '__main__':
    main()
