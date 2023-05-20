import numpy as np


def likelihood(p_vectors, query, collection=False, eps=1e-5):
    query_array = (query != 0).reshape(-1)

    if not query_array.any():
        return np.zeros((len(p_vectors), 1))

    if collection:
        if p_vectors.sum() != 0:
            p_vectors = np.asarray(p_vectors.sum(axis=0) / p_vectors.sum())
            return p_vectors[query_array] + eps

        return np.asarray(p_vectors.sum(axis=0)[query_array]) + eps

    return np.asarray(p_vectors[:, query_array]) + eps


class RetrievalSystem(object):
    def __init__(self):
        self.corpus = None

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

        if type(corpus) != np.ndarray:
            corpus = corpus.toarray()

        self.corpus = corpus

    def predict(self, queries):
        """Return the similarity scores associated with each document in the corpus for each query.

        Parameters
        ----------
        queries : List
            List of queries
            Each query is a TF-IDF embedding

        Returns
        -------
        ndarray (n_queries, n_documents)
            Similarity scores associated with each document in the corpus for each query
        """

        if type(queries) != np.ndarray:
            queries = queries.toarray()

        predictions = np.zeros((queries.shape[0], self.corpus.shape[0]))

        for i, query in enumerate(queries):
            predictions[i] = self.eval_query(query)

        return predictions

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

        pass
