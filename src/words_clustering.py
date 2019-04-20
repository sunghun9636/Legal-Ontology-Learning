import pickle
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

from clusters_evaluation import pca_topics_visualization, agglomerative_clusters_silhouette_score
from data_preparation import remove_low_high_frequent_words, extract_important_words_tfidf
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def get_corpus_words(data):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)
    dictionary = Dictionary(documents)

    return dictionary.values()  # returning all words in the data corpus


def words_to_self_trained_word2vec(train_data, words):
    with open(train_data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)

    # train word2vec model with the documents
    word2vec_model = Word2Vec(documents, size=100, window=10, min_count=1, workers=10)

    vectors = [word2vec_model.wv[word] for word in words]
    return vectors


def corpus_words_clustering(words_in_vectors):
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    predictions = cluster.fit_predict(words_in_vectors)

    pca = PCA(n_components=3)  # 3 dimension PCA model
    pca.fit(words_in_vectors)
    data_pca = pca.transform(words_in_vectors)

    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.scatter3D(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=cluster.labels_, cmap='rainbow')
    plt.show()


def main():
    words = get_corpus_words('data/case_documents_20000.data')
    words_in_vectors = words_to_self_trained_word2vec('data/case_documents_20000.data', words)
    # pca_topics_visualization(words_in_vectors, "")

    corpus_words_clustering(words_in_vectors)

    # ++++++++++++++ Agglomerative Hierarchical clustering evaluation ++++++ #
    # clustering_methods = ['ward', 'complete', 'average', 'single']  # Agglomerative Clustering Methods
    for method in ['ward']:
        for n_clusters in range(2, 10):  # 2 <= n_clusters <= n_topics(10) - 1
            agglomerative_clusters_silhouette_score(words_in_vectors,
                                                    n_clusters,
                                                    method)


if __name__ == '__main__':
    main()
