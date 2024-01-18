import numpy as np
from scipy.special import logsumexp, binom
from itertools import chain, combinations
from functools import lru_cache

class ScorePruner:
    def __init__(self, _scores, _K, _eps, _b):
        self.scores = _scores
        self.K = _K
        self.beta_R = 1 / _K
        self.eps = _eps
        self.b = _b

    def prune_scores(self):
        orig_count = self.count_scores()
        vertices = list(self.scores.keys())

        for i in vertices:
            self.prune_scores_node(i)

        pruned_count = self.count_scores()
        print(f'b={self.b} From {orig_count} to {pruned_count}')

    def count_scores(self):
        return sum(len(s) for s in self.scores.values())

    @lru_cache(maxsize=None)
    def psi(self, i, j, S):
        Rs = np.array([frozenset(subset)
                       for subset in chain.from_iterable(combinations(S, r)
                                                         for r in range(len(S)+1))
                       if j in subset])
        res = [self.pi(i, R) + np.log(self.w(R, S)) for R in Rs]
        return logsumexp(res)

    def w(self, R, S):
        return (1 + self.beta_R) ** (len(R) - self.K) * (self.beta_R) ** (len(S) - len(R))

    @lru_cache(maxsize=None)
    def pi(self, v, pa_i):
        k = len(pa_i)
        n = len(self.scores.keys())

        try:
            res = self.scores[v][pa_i]
            prior = np.log(1 / binom(n, k))
            res += prior
        except KeyError:
            res = -np.inf

        return res * self.b

    def prune_scores_node(self, i):
        for S in list(self.scores[i].keys()):
            if not S:
                continue

            is_prune = all(self.pi(i, S) < (np.log(self.eps) + self.psi(i, j, S)) for j in S)
            if is_prune:
                del self.scores[i][S]
