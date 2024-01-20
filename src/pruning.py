import numpy as np
from scipy.special import logsumexp, binom
from itertools import chain, combinations
from functools import lru_cache
import copy

class ScorePruner:
    def __init__(self, _scores, _K, _eps, _b):
        self.scores = copy.deepcopy(_scores)
        self.K = _K
        self.beta_R = 1 / _K
        self.log_eps = np.log(_eps)
        self.b = _b

    def prune_scores(self):
        orig_count = self.count_scores()
        vertices = list(self.scores.keys())

        for i in vertices:
            self.prune_scores_node(i)

        pruned_count = self.count_scores()
        print(f'b={self.b} From {orig_count} to {pruned_count}')

        return self.scores

    def count_scores(self):
        return sum(len(s) for s in self.scores.values())

    @lru_cache(maxsize=None)
    def psi(self, i, j, S):
        S_without_j = set(S) - {j}
        subsets_including_j = [frozenset({j} | set(subset))
                            for r in range(len(S))
                            for subset in combinations(S_without_j, r)]

        pi_values = np.array([self.pi(i, R) for R in subsets_including_j])
        w_values = np.array([self.w(R, S) for R in subsets_including_j])

        return logsumexp(pi_values + np.log(w_values))

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
        keys_to_delete = []

        for S in list(self.scores[i].keys()):
            if not S:
                continue
            
            pi_S = self.pi(i, S)
            is_prune = all(pi_S < (self.log_eps + self.psi(i, j, S)) for j in S)
            if is_prune:
                keys_to_delete.append(S)

        for key in keys_to_delete:
            del self.scores[i][key]