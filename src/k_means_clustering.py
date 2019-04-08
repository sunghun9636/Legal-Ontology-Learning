import pickle
import numpy as np
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from topic_model import train_lda_model
from data_preparation import remove_low_high_frequent_words


def get_lda_topics(lda_model):  # return list of (list of terms per topic) from lda model

    topics = lda_model.show_topics(num_words=20, formatted=False)
    topics_words = [[word[0] for word in topic[1]] for topic in topics]

    return topics_words


# ++++++++++++++++++++++++++++++ Word Embeddings Algorithms +++++++++++++++++++++++++++++++++++ #


def sent_vectorizer(sent, model):  # map sentence into vector by averaging words' embeddings from the "model"
    sent_vec = []
    num_words = 0
    for w in sent:
        try:
            if num_words == 0:
                sent_vec = model.wv[w]
            else:
                sent_vec = np.add(sent_vec, model.wv[w])
            num_words += 1
        except:
            pass

    return np.asarray(sent_vec) / num_words


# transforming topic words to word embedding using self trained word2vec from documents
def self_trained_word2vec(training_corpus, topics_words):
    with open(training_corpus, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)

    # train word2vec model with the documents
    word2vec_model = Word2Vec(documents, size=100, window=10, min_count=1, workers=10)

    topics_in_vector = []
    for terms_list in topics_words:
        topics_in_vector.append(sent_vectorizer(terms_list, word2vec_model))

    return topics_in_vector  # list of vectors (vector per topic)


def self_trained_doc2vec(training_corpus, topics_words):
    with open(training_corpus, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)
    documents = [" ".join(doc) for doc in documents]

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    doc2vec_model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=5)

    topics_in_vector = []
    for terms_list in topics_words:
        topics_in_vector.append(doc2vec_model.infer_vector(terms_list))

    return topics_in_vector

# ---------------------------------------------------------------------------------------------- #
# ++++++++++++++++++++++++++++++ Hierarchical K-means clustering algorithm +++++++++++++++++++++ #


def dendrogram(data):
    plt.figure(figsize=(10, 7))
    plt.title("Topics Dendrogram")
    dendrogram = shc.dendrogram(shc.linkage(data, method='centroid'))  # method='centroid' for k-means clustering
    plt.show()

# ---------------------------------------------------------------------------------------------- #


def main():
    lda_model = train_lda_model('data/case_documents_5000.data', 10)[0]
    topic_words = get_lda_topics(lda_model)

    # ++++++++++++ LDA topics in vector using self trained word2vec +++++++++++++ #
    lda_topics_in_vectors = self_trained_word2vec('data/case_documents_5000.data', topic_words)
    print(lda_topics_in_vectors)
    # --------------------------------------------------------------------------- #

    # ++++++++++++ LDA topics in vector using self trained doc2vec +++++++++++++ #
    # lda_topics_in_vectors = self_trained_doc2vec('data/case_documents_5000.data', topic_words)
    # print(lda_topics_in_vectors)
    # --------------------------------------------------------------------------- #

    # ++++++++++++++ Hierarchical K-means clustering algorithm +++++++++++++++++++++ #
    dendrogram(lda_topics_in_vectors)  # output dendrogram diagram


if __name__ == '__main__':
    main()
