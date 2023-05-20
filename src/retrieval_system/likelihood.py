from .base import RetrievalSystem, likelihood
import numpy as np


class LikelihoodRetrieval(RetrievalSystem):
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

        return likelihood(self.corpus, query).prod(axis=1)
