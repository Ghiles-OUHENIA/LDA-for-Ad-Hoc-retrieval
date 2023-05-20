from .base import RetrievalSystem, likelihood
import numpy as np


def init_params(n_docs, n_topics, vocab_len):
    psi = np.zeros((n_topics, vocab_len))
    theta = np.zeros((n_docs, n_topics))
    return psi, theta


def sample_topic(doc_id, word_id, psi, theta, alpha, beta):
    n_topics, n_words = psi.shape
    topic_probs = (psi[:, word_id] + beta) / (psi.sum(axis=1) + beta * n_words)
    doc_probs = (theta[doc_id, :] + alpha) / (
        theta.sum(axis=1)[doc_id] + alpha * n_topics
    )

    p_choice = topic_probs * doc_probs
    p_choice /= p_choice.sum()

    new_topic = np.random.choice(np.arange(n_topics), p=p_choice)

    psi[new_topic, word_id] += 1
    theta[doc_id, new_topic] += 1


def gibbs_sampling(docs, n_topics, n_iter, nb_mc, alpha, beta):
    n_docs, vocab_len = docs.shape

    psi_markov = np.zeros((n_topics, vocab_len))
    theta_markov = np.zeros((n_docs, n_topics))

    for _ in range(nb_mc):
        psi, theta = init_params(n_docs, n_topics, vocab_len)

        for _ in range(n_iter):
            for doc_id, doc in enumerate(docs):
                for word_id in np.where(doc != 0)[0]:
                    sample_topic(doc_id, word_id, psi, theta, alpha, beta)

        psi_markov += psi
        theta_markov += theta

    psi_markov = (psi_markov + beta) / (
        psi_markov.sum(axis=1) + beta * vocab_len
    ).reshape(-1, 1)
    theta_markov = (theta_markov + alpha) / (
        theta_markov.sum(axis=1) + alpha * n_topics
    ).reshape(-1, 1)

    return psi_markov, theta_markov


class LdaRetrieval(RetrievalSystem):
    def __init__(
        self,
        n_topics,
        mu=1000,
        alpha=None,
        beta=0.01,
        n_iter=1,
        nb_mc=1,
        lmbda=0.7,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.mu = mu
        self.Nds = None
        self.alpha = alpha or 50 / n_topics
        self.beta = beta
        self.n_iter = n_iter
        self.nb_mc = nb_mc
        self.lmbda = lmbda

        self.gibbs_estimator = None

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

        psi, theta = gibbs_sampling(
            self.corpus,
            self.n_topics,
            n_iter=self.n_iter,
            nb_mc=self.nb_mc,
            alpha=self.alpha,
            beta=self.beta,
        )

        self.gibbs_estimator = theta @ psi

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

        n_docs, len_vocab = self.corpus.shape
        P_w_D = np.zeros(n_docs)

        ql_doc = likelihood(self.corpus, query)
        ql_coll = likelihood(self.corpus, query, collection=True)

        query_indices = np.where(query != 0)[0]

        for d in range(n_docs):
            Nd = self.Nds[d]
            P_w_D[d] = (
                self.lmbda
                * (
                    (Nd / (Nd + self.mu)) * ql_doc[d]
                    + (1 - (Nd / (Nd + self.mu))) * ql_coll
                )
                + (1 - self.lmbda) * self.gibbs_estimator[d, query_indices]
            ).prod()

        return P_w_D
