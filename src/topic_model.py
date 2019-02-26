import gensim.models as models
import pickle
from data_preparation import remove_low_high_frequent_words, get_tfidf
import pyLDAvis.gensim


def train_lda_model(data, num_topics):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    in2word = get_tfidf(documents)["index2word"]

    lda_model = models.ldamodel.LdaModel(corpus=corpus,
                                         id2word=in2word,
                                         num_topics=num_topics,
                                         passes=10)

    return lda_model, corpus, in2word


def train_svd_model(data, num_topics):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    in2word = get_tfidf(documents)["index2word"]

    svd_model = models.LsiModel(corpus=corpus,
                                id2word=in2word,
                                num_topics=num_topics,
                                chunksize=5,
                                onepass=False,
                                power_iters=10)

    return svd_model, corpus, in2word


def lda_visualization(data, num_topics):
    model, corpus, dictionary = train_lda_model(data, num_topics)
    print("length of the dictionary is: ")
    print(len(dictionary))

    visual = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(visual, 'visual/lda_visual.html')


def main():
    # print("RESULT FROM LDA: ")
    # print(train_lda_model('data/case_documents_1000.data', 10).print_topics())
    # print("RESULT FROM SVD: ")
    # print(train_svd_model('data/case_documents_1000.data', 10).print_topics())

    # LDA visualization
    lda_visualization('data/case_documents_1000.data', 10)


if __name__ == '__main__':
    main()
