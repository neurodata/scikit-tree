# XXX: trying to sketch out a base causal API that extends "BaseEstimator" of sklearn
class BaseCausal:
    def conditional_effect(self, X=None, *, T0, T1):
        pass

    def marginal_effect(self, T, X=None):
        pass
