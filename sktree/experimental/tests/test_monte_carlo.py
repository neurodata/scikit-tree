import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

from sktree.experimental.monte_carlo import conditional_resample


def test_conditional_resample():
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Convert X to sparse matrix
    X_sparse = csr_matrix(X)

    # Create conditional array
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    conditional_array = nn.kneighbors_graph(X).toarray()

    # Perform conditional resampling
    resampled_X = conditional_resample(conditional_array, X, replace=False, replace_nbrs=False)
    resampled_X_sparse = conditional_resample(
        conditional_array, X_sparse, replace=False, replace_nbrs=False
    )

    # Check that the resampled arrays have the correct shape
    assert resampled_X.shape == X.shape
    assert resampled_X_sparse.shape == X_sparse.shape

    # Check that the resampled arrays have the correct number of unique samples
    assert len(np.unique(resampled_X, axis=0)) == X.shape[0]
    assert len(np.unique(resampled_X_sparse.toarray(), axis=0)) == X_sparse.shape[0]

    # Check that the conditional distribution is preserved
    for i in range(X.shape[1]):
        unique_values, counts = np.unique(resampled_X[:, i], return_counts=True)
        original_values, original_counts = np.unique(X[:, i], return_counts=True)

        assert np.all(unique_values == original_values)
        assert np.all(counts == original_counts)

        unique_values_sparse, counts_sparse = np.unique(
            resampled_X_sparse[:, i].toarray(), return_counts=True
        )
        original_values_sparse, original_counts_sparse = np.unique(
            X_sparse[:, i].toarray(), return_counts=True
        )

        assert np.all(unique_values_sparse == original_values_sparse)
        assert np.all(counts_sparse == original_counts_sparse)
