import gensim.models as models
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    print('{} {}'.format('length of the dictionary is: ', len(dictionary)))

    visual = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(visual, 'visual/lda_visual.html')


def plot_word_importance(model):
    plt.figure(figsize=(15, 30))

    for i in range(model.get_topics().shape[0]):  # number of topics
        df = pd.DataFrame(model.show_topic(i), columns=['term', 'prob']).set_index('term')

        plt.subplot(model.get_topics().shape[0]/2, 2, i + 1)  # two plots per row
        plt.title('topic ' + str(i + 1))
        sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Reds_d')
        plt.xlabel('probability')

    plt.show()


def main():
    # print("RESULT FROM LDA: ")
    # print(train_lda_model('data/case_documents_1000.data', 10).print_topics())
    # print("RESULT FROM SVD: ")
    # print(train_svd_model('data/case_documents_1000.data', 10).print_topics())

    # ------------- LDA visualization ---------------- #
    # lda_visualization('data/case_documents_2500.data', 10)

    # ------------- LDA word importance visualization ---------------- #
    lda_model = train_lda_model('data/case_documents_2500.data', 10)[0]
    plot_word_importance(lda_model)


if __name__ == '__main__':
    main()
