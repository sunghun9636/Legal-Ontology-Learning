import pickle
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

from clusters_evaluation import pca_topics_visualization
from data_preparation import remove_low_high_frequent_words


def get_corpus_words(data):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)
    dictionary = Dictionary(documents)

    return dictionary.values()  # returning all words in the data corpus


def words_to_self_trained_word2vec(train_data, words):
    with open(train_data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)  # training corpus for word2vec model

    # train word2vec model with the documents
    word2vec_model = Word2Vec(documents, size=100, window=10, min_count=1, workers=10)

    vectors = [word2vec_model.wv[word] for word in words]
    return vectors


def main():
    words = get_corpus_words('data/case_documents_20000.data')
    words_in_vectors = words_to_self_trained_word2vec('data/case_documents_20000.data', words)
    pca_topics_visualization(words_in_vectors, "")


if __name__ == '__main__':
    main()
