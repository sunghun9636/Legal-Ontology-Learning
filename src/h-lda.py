import pickle
from gensim.corpora import Dictionary
from hlda.sampler import HierarchicalLDA
from data_preparation import remove_low_high_frequent_words


def train_hlda_model(data):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)

    # ++++++ preparing dictionary and corpus for l-LDA library ++++++ #
    dictionary_set = set()
    for doc in documents:
        for word in doc:
            dictionary_set.add(word)

    vocab = sorted(list(dictionary_set))
    vocab_index = {}
    for i, w in enumerate(vocab):
        vocab_index[w] = i

    new_corpus = []
    for doc in documents:
        new_doc = []
        for word in doc:
            word_idx = vocab_index[word]
            new_doc.append(word_idx)
        new_corpus.append(new_doc)
    # --------------------------------------------------------------- #

    hlda_model = HierarchicalLDA(corpus=new_corpus,
                                 vocab=vocab,
                                 alpha=10.0,  # default = 10.0
                                 gamma=1.0,  # default = 1.0
                                 eta=1.0,  # default = 0.1
                                 seed=0,  # default = 0
                                 verbose=True,  # default = True
                                 num_levels=3)  # default = 3

    return hlda_model, new_corpus, vocab_index


def main():

    # ------------- h-LDA run sampler ---------------- #
    hlda_model = train_hlda_model('data/case_documents_5000.data')[0]

    hlda_model.estimate(num_samples=100,  # default = 500
                        display_topics=10,  # default = 50
                        n_words=5,  # default = 5
                        with_weights=False)  # default = True


if __name__ == '__main__':
    main()
