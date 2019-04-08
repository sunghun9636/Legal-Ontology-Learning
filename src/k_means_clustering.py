from topic_model import train_lda_model
import pandas as pd


def get_lda_topics(lda_model):  # return list of (list of terms per topic) from lda model

    topics = lda_model.show_topics(num_words=20, formatted=False)
    topics_words = [[word[0] for word in topic[1]] for topic in topics]

    return topics_words


def main():
    lda_model = train_lda_model('data/case_documents_5000.data', 10)[0]
    topic_words = get_lda_topics(lda_model)
    print(topic_words)


if __name__ == '__main__':
    main()
