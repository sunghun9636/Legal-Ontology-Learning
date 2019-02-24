import gensim.models as models
import pickle
from data_preparation import remove_low_high_frequent_words, get_tfidf
import pyLDAvis.gensim


def train_lda_model(data, num_topics):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.10, 0.30)

    lda_model = models.ldamodel.LdaModel(corpus=get_tfidf(documents)['corpus_tfidf'],
                                         id2word=get_tfidf(documents)['index2word'],
                                         num_topics=num_topics,
                                         passes=10)

    return lda_model, get_tfidf(documents)['corpus_tfidf'], get_tfidf(documents)['index2word']


def train_svd_model(data, num_topics):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.10, 0.50)

    svd_model = models.LsiModel(corpus=get_tfidf(documents)['corpus_tfidf'],
                                id2word=get_tfidf(documents)['index2word'],
                                num_topics=num_topics,
                                chunksize=5,
                                onepass=False,
                                power_iters=10)

    return svd_model


def lda_visualization():
    model, corpus, dictionary = train_lda_model('data/case_documents_1000.data', 10)

    visual = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(visual, 'visual.html')


def main():
    # print("RESULT FROM LDA: ")
    # print(train_lda_model('data/case_documents_100.data', 10).print_topics())
    # print("RESULT FROM SVD: ")
    # print(train_svd_model('data/case_documents_100.data', 10).print_topics())

    # visualization
    lda_visualization()


if __name__ == '__main__':
    main()
