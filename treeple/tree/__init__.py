from .._lib.sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from ._classes import (
    ExtraObliqueDecisionTreeClassifier,
    ExtraObliqueDecisionTreeRegressor,
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeRegressor,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)
from ._honest_tree import HonestTreeClassifier
from ._kernel import KernelDecisionTreeClassifier
from ._multiview import MultiViewDecisionTreeClassifier
from ._neighbors import compute_forest_similarity_matrix

__all__ = [
    "ExtraObliqueDecisionTreeClassifier",
    "ExtraObliqueDecisionTreeRegressor",
    "compute_forest_similarity_matrix",
    "UnsupervisedDecisionTree",
    "UnsupervisedObliqueDecisionTree",
    "ObliqueDecisionTreeClassifier",
    "ObliqueDecisionTreeRegressor",
    "PatchObliqueDecisionTreeClassifier",
    "PatchObliqueDecisionTreeRegressor",
    "HonestTreeClassifier",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "MultiViewDecisionTreeClassifier",
    "KernelDecisionTreeClassifier",
]
