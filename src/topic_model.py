import gensim.models as models
import pickle
from data_preparation import remove_low_frequent_words, get_tfidf


def train_lda_model(num_topics):
    with open('case_documents.data', 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_frequent_words(documents, 0.10)

    lda_model = models.ldamodel.LdaModel(corpus=get_tfidf(documents)['corpus_tfidf'],
                                         id2word=get_tfidf(documents)['index2word'],
                                         num_topics=num_topics,
                                         random_state=100,
                                         update_every=1,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         per_word_topics=True)
    return lda_model


def main():
    print(train_lda_model(10).print_topics())


if __name__ == '__main__':
    main()
