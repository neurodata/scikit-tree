from time import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sktree import MultiViewRandomForestClassifier

seed = 12345
rng = np.random.default_rng(seed)

n_repeats = 5
n_jobs = -1
n_estimators = 6000
n_samples = 256
n_dims = 1000
X = rng.standard_normal(size=(n_samples, n_dims))
y = rng.integers(0, 2, size=(n_samples,))


# for idx in range(n_repeats):
#     clf = RandomForestClassifier(
#         n_estimators=n_estimators, max_features=0.3, random_state=seed, n_jobs=n_jobs
#     )
#     tstart = time()
#     clf.fit(X, y)
#     fit_time = time() - tstart
#     print(f"Fit time for RandomForestClassifier: {fit_time}")


for idx in range(n_repeats):
    mv_clf = MultiViewRandomForestClassifier(
        n_estimators=n_estimators,
        feature_set_ends=[n_dims // 2, n_dims],
        max_features=[0.3, 0.3],
        random_state=seed,
        n_jobs=n_jobs,
    )
    tstart = time()
    mv_clf.fit(X, y)
    fit_time = time() - tstart
    print(f"Fit time for MultiViewRandomForestClassifier: {fit_time}")
