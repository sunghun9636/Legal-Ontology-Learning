import pickle
import random
import numpy as np
import gensim.models as models
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from data_preparation import remove_low_high_frequent_words, get_tfidf, extract_important_words_tfidf


def save_train_and_test(data):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    dictionary = get_tfidf(documents)["index2word"]

    corpus = list(corpus)

    random.shuffle(corpus)
    train = corpus[:18000]
    test = corpus[18000:]

    with open('data/train_corpus.data', 'wb') as file:
        print("...Saving training corpus into local binary file...")
        pickle.dump(train, file)

    with open('data/test_corpus.data', 'wb') as file:
        print("...Saving test corpus into local binary file...")
        pickle.dump(test, file)

    with open('data/common_dictionary.data', 'wb') as file:
        print("...Saving common dictionary for train and test corpus into binary file...")
        pickle.dump(dictionary, file)


def perplexity(data, limit, start=2, step=1):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    dictionary = get_tfidf(documents)["index2word"]

    # random.shuffle(corpus)
    train = corpus[:15000]
    test = corpus[15000:]

    perplexity_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = models.ldamodel.LdaModel(corpus=train, id2word=dictionary, num_topics=num_topics, eta=0.3)
        model_list.append(model)

        perplexity_values.append(model.bound(test))

    # model = models.ldamodel.LdaModel(corpus=train, id2word=dictionary, num_topics=5, eta=0.3)
    #
    # perplexity_score = model.log_perplexity(test)
    #
    # print('{} {}'.format('Perplexity score is: ', perplexity_score))

    return model_list, perplexity_values


def coherence(data, limit, start=2, step=1):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    dictionary = get_tfidf(documents)["index2word"]

    # random.shuffle(corpus)
    train = corpus[:15000]
    test = corpus[15000:]

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = models.ldamodel.LdaModel(corpus=train, id2word=dictionary, num_topics=num_topics, eta=0.3)
        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, dictionary=dictionary, corpus=test, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def main():
    # save_train_and_test(data='data/case_documents_20000.data')

    limit = 15
    start = 2
    step = 1
    model_list, coherence_values = coherence(data='data/case_documents_20000.data', limit=limit, start=start, step=step)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score")
    plt.legend("coherence_score", loc='best')
    plt.show()

    # model_list, perplexity_values = perplexity(data='data/case_documents_20000.data', limit=15, start=2, step=1)
    # limit = 15
    # start = 2
    # step = 1
    # x = range(start, limit, step)
    # plt.plot(x, perplexity_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Perplexity Score")
    # plt.legend("Perplexity", loc='best')
    # plt.show()


if __name__ == '__main__':
    main()
