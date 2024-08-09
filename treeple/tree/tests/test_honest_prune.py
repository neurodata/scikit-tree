import numpy as np

from treeple.tree import HonestTreeClassifier


def test_honest_tree_pruning():
    """Test honest tree with pruning to ensure no empty leaves."""
    rng = np.random.default_rng(1234)

    n_samples = 1000
    X = rng.standard_normal(size=(n_samples, 100))
    X[n_samples // 2 :] *= -1
    y = [0] * (n_samples // 2) + [1] * (n_samples // 2)

    clf = HonestTreeClassifier(honest_method="prune", max_features="sqrt", random_state=0)
    clf = clf.fit(X, y)

    nonprune_clf = HonestTreeClassifier(
        honest_method="apply", max_features="sqrt", random_state=0, honest_prior="ignore"
    )
    nonprune_clf = nonprune_clf.fit(X, y)

    assert (
        nonprune_clf.tree_.max_depth >= clf.tree_.max_depth
    ), f"{nonprune_clf.tree_.max_depth} <= {clf.tree_.max_depth}"
    # assert np.all(clf.tree_.children_left != -1)

    # Access the original and pruned trees' attributes
    original_tree = nonprune_clf.tree_
    pruned_tree = clf.tree_

    # Ensure the pruned tree has fewer or equal nodes
    assert (
        pruned_tree.node_count < original_tree.node_count
    ), "Pruned tree has more nodes than the original tree"

    # Ensure the pruned tree has no empty leaves
    assert np.all(pruned_tree.value.sum(axis=(1, 2)) > 0), pruned_tree.value.sum(axis=(1, 2))
    # assert np.all(original_tree.value.sum(axis=(1,2)) > 0), original_tree.value.sum(axis=(1,2))
    assert np.all(pruned_tree.value.sum(axis=(1, 2)) > 0) > np.all(
        original_tree.value.sum(axis=(1, 2)) > 0
    )

    # test that the first three nodes are the same, since these are unlikely to be
    # pruned, and should remain invariant.
    #
    # Note: pruning the tree will have the node_ids change since the tree is
    # ordered via DFS.
    for pruned_node_id in range(3):
        pruned_left_child = pruned_tree.children_left[pruned_node_id]
        pruned_right_child = pruned_tree.children_right[pruned_node_id]

        # Check if the pruned node exists in the original tree
        assert (
            pruned_left_child in original_tree.children_left
        ), "Left child node of pruned tree not found in original tree"
        assert (
            pruned_right_child in original_tree.children_right
        ), "Right child node of pruned tree not found in original tree"

        # Check if the node's parameters match for non-leaf nodes
        if pruned_left_child != -1:
            assert (
                pruned_tree.feature[pruned_node_id] == original_tree.feature[pruned_node_id]
            ), "Feature does not match for node {}".format(pruned_node_id)
            assert (
                pruned_tree.threshold[pruned_node_id] == original_tree.threshold[pruned_node_id]
            ), "Threshold does not match for node {}".format(pruned_node_id)
            assert (
                pruned_tree.weighted_n_node_samples[pruned_node_id]
                == original_tree.weighted_n_node_samples[pruned_node_id]
            ), "Weighted n_node samples does not match for node {}".format(pruned_node_id)
