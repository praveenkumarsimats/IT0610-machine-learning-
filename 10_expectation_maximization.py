"""
Program 10: Expectation & Maximization (EM) Algorithm
Implements EM-based clustering using a Gaussian Mixture Model (GMM) and
compares the clustering result with K-Means, on the Iris dataset.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y_true = iris.target

    # --- K-Means clustering ---
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_ari = adjusted_rand_score(y_true, kmeans_labels)

    # --- EM algorithm via Gaussian Mixture Model ---
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    gmm_ari = adjusted_rand_score(y_true, gmm_labels)

    print("=== K-Means Clustering ===")
    print(f"Adjusted Rand Index (vs true labels): {kmeans_ari:.4f}")

    print("\n=== EM Algorithm (Gaussian Mixture Model) ===")
    print(f"Adjusted Rand Index (vs true labels): {gmm_ari:.4f}")
    print(f"Converged: {gmm.converged_}, Iterations run: {gmm.n_iter_}")
    print("\nLearned Gaussian means (per component):")
    print(gmm.means_)

    print("\nConclusion:")
    if gmm_ari > kmeans_ari:
        print("EM/GMM clustering matched the true species labels better than K-Means.")
    else:
        print("K-Means clustering matched the true species labels better than EM/GMM "
              "on this run.")

    # Visualize using first two features (sepal length & width)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis")
    axes[0].set_title("True Labels")

    axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap="viridis")
    axes[1].set_title("K-Means Clusters")

    axes[2].scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap="viridis")
    axes[2].set_title("EM (GMM) Clusters")

    for ax in axes:
        ax.set_xlabel("Sepal length")
        ax.set_ylabel("Sepal width")

    plt.tight_layout()
    plt.savefig("em_vs_kmeans_clustering.png")
    print("\nPlot saved as 'em_vs_kmeans_clustering.png'")
