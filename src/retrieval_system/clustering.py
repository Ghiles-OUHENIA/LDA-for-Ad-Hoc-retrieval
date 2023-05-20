from .base import RetrievalSystem, likelihood
import numpy as np
from sklearn.cluster import KMeans


class ClusterBasedRetrieval(RetrievalSystem):
    def __init__(self, n_topics, mu=1000):
        super().__init__()

        self.doc_clusters = None
        self.clusters = None

        self.Nds = None

        self.n_topics = n_topics
        self.mu = mu

    def fit(self, corpus, **kwargs):
        """Train the retrieval system on the corpus

        Parameters
        ----------
        corpus : List(array-like)
            The corpus of documents
            Each document is represented by a TF-IDF embedding (frequencies)

        Returns
        -------
        None
        """

        super().fit(corpus, **kwargs)

        self.Nds = kwargs.get("Nds", self.Nds)

        k_means = KMeans(n_clusters=self.n_topics, verbose=False, random_state=0)
        k_means.fit(self.corpus)

        self.doc_clusters = k_means.labels_
        self.clusters = np.unique(self.doc_clusters)

    def eval_query(self, query):
        """
        Evaluate the query and calculate the similarity scores for each document in the corpus.

        Parameters
        ----------
        query : array-like
            Query TF-IDF embedding

        Returns
        -------
        ndarray (n_documents,)
            Similarity scores for each document in the corpus
        """

        n_docs = self.corpus.shape[0]
        P_w_D = np.zeros(n_docs)

        ql_doc = likelihood(self.corpus, query)

        ql_cluster = dict()

        for cl in self.clusters:
            vectors_cluster = self.corpus[self.doc_clusters == cl]
            ql_cluster[cl] = likelihood(vectors_cluster, query, collection=True)

        for d in range(n_docs):
            Nd = self.Nds[d]
            P_w_D[d] = (
                (Nd / (Nd + self.mu)) * ql_doc[d]
                + (1 - (Nd / (Nd + self.mu))) * ql_cluster[self.doc_clusters[d]]
            ).prod()

        return P_w_D
