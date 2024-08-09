"""
======================================
Custom Kernel Decision Tree Classifier
======================================

This example shows how to build a manifold oblique decision tree classifier using
a custom set of user-defined kernel/filter library, such as the Gaussian, or Gabor
kernels.

The example demonstrates superior performance on a 2D dataset with structured images
as samples. The dataset is the downsampled MNIST dataset, where each sample is a
28x28 image. The dataset is downsampled to 14x14, and then flattened to a 196
dimensional vector. The dataset is then split into a training and testing set.

See :ref:`sphx_glr_auto_examples_plot_projection_matrices` for more information on
projection matrices and the way they can be sampled.
"""

import matplotlib.pyplot as plt

# %%
# Importing the necessary modules
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from treeple.tree import KernelDecisionTreeClassifier

# %%
# Load the Dataset
# ----------------
# We need to load the dataset and split it into training and testing sets.

# Load the dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

# Downsample the dataset
X = X.reshape((-1, 28, 28))
X = X[:, ::2, ::2]
X = X.reshape((-1, 196))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Setting up the Custom Kernel Decision Tree Model
# -------------------------------------------------
# To set up the custom kernel decision tree model, we need to define typical hyperparameters
# for the decision tree classifier, such as the maximum depth of the tree and the minimum
# number of samples required to split an internal node. For the Kernel Decision tree model,
# we also need to define the kernel function and its parameters.

max_depth = 10
min_samples_split = 2

# Next, we define the hyperparameters for the custom kernels that we will use.
# For example, if we want to use a Gaussian kernel with a sigma of 1.0 and a size of 3x3:
kernel_function = "gaussian"
kernel_params = {"sigma": 1.0, "size": (3, 3)}

# We can then fit the custom kernel decision tree model to the training set:
clf = KernelDecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    data_dims=(28, 28),
    min_patch_dims=(1, 1),
    max_patch_dims=(14, 14),
    dim_contiguous=(True, True),
    boundary=None,
    n_classes=10,
    kernel_function=kernel_function,
    n_kernels=500,
    store_kernel_library=True,
)

# Fit the decision tree classifier using the custom kernel
clf.fit(X_train, y_train)

# %%
# Evaluating the Custom Kernel Decision Tree Model
# ------------------------------------------------
# To evaluate the custom kernel decision tree model, we can use the testing set.
# We can also inspect the important kernels that the tree selected.

# Predict the labels for the testing set
y_pred = clf.predict(X_test)

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print(f"Kernel decision tree model obtained an accuracy of {accuracy} on MNIST.")

# Get the important kernels from the decision tree classifier
important_kernels = clf.kernel_arr_
kernel_dims = clf.kernel_dims_
kernel_params = clf.kernel_params_
kernel_library = clf.kernel_library_

# Plot the important kernels
fig, axes = plt.subplots(
    nrows=len(important_kernels), ncols=1, figsize=(6, 4 * len(important_kernels))
)
for i, kernel in enumerate(important_kernels):
    axes[i].imshow(kernel, cmap="gray")
    axes[i].set_title("Kernel {}".format(i + 1))
plt.show()
