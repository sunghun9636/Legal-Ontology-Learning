import pyLDAvis
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from k_means_clustering import get_lda_topics, self_trained_word2vec, dendrogram
from topic_model import train_lda_model


def agglomerative_clusters_silhouette_score(data, n_clusters, linkage):  # agglomerative for hierarchical clustering

    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage)
    predictions = cluster.fit_predict(data)

    score = silhouette_score(data, predictions, metric='euclidean')  # silhouette score for the clusters

    print("For n_clusters = {}, linkage = {}, Silhouette Score is {})".format(n_clusters, linkage, score))


def main():
    lda_model, corpus, dictionary = train_lda_model('data/case_documents_20000.data', 10)  # LDA topic modelling
    visual = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)  # saving visual presentation of topics
    pyLDAvis.save_html(visual, 'visual/lda_visual.html')

    topic_words = get_lda_topics(lda_model)  # getting the top topic words
    print(topic_words)

    # ++++++++++++ LDA topics in vector using self trained word2vec +++++++++++++ #
    lda_topics_in_vectors = self_trained_word2vec('data/case_documents_20000.data', topic_words)
    # print(lda_topics_in_vectors)
    # --------------------------------------------------------------------------- #

    # ++++++++++++++ Hierarchical clustering algorithm +++++++++++++++++++++ #
    dendrogram(lda_topics_in_vectors, 'ward')  # output dendrogram diagram

    # ++++++++++++++ Hierarchical clustering evaluation +++++++++++++++++++++ #
    for n_clusters in range(2, 10):  # 2 <= n_clusters <= n_topics -1
        agglomerative_clusters_silhouette_score(lda_topics_in_vectors,
                                                n_clusters,
                                                'ward')


if __name__ == '__main__':
    main()
