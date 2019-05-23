import pickle
import random
import gensim.models as models
from data_preparation import remove_low_high_frequent_words, get_tfidf, extract_important_words_tfidf


def perplexity(data):
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

    model = models.ldamodel.LdaModel(corpus=train, id2word=dictionary, num_topics=5, eta=0.3)

    perplexity_score = model.log_perplexity(test)

    print('{} {}'.format('Perplexity score is: ', perplexity_score))


def main():
    perplexity('data/case_documents_20000.data')


if __name__ == '__main__':
    main()
