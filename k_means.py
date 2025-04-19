import os
import numpy as np
import matplotlib.pyplot as plt


class K_means:
    def __init__(self, k, domain = [-5, 5]):
        """
        Initializes K_means clustering.

        Inputs:
            - k (int): number of clusters
            - domain (list): range of values for the data points
        """
        self.k = k
        self.domain = domain
        self.dim = 2

        self.iteration = 0

        self.mu = self._initialize_parameters()


        self.images_dir = "images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        else:
            # remove all images in the directory
            for file in os.listdir(self.images_dir):
                if file.endswith(".png"):
                    os.remove(os.path.join(self.images_dir, file))


    def _initialize_parameters(self):
        """
        Initialize the parameters of the MoG model
        """
        # initialize means randomly within the domain
        self.mu = np.random.rand(self.k, self.dim) * (self.domain[1] - self.domain[0]) + self.domain[0]

        return self.mu
    
    def _distances(self, mu):
        """
        Computes the Euclidean distance between each point in X and each mean in mu

        Inputs:
            - mu (array-like): shape (n_clusters, n_features)

        Returns:
            - distances (ndarray): shape (n_samples, n_clusters)
        """
        return np.linalg.norm(self.data[:, np.newaxis] - mu, axis=2)
    
    def init_k_means_algorithm(self, X):
        """
        Initialize the EM algorithm with the given data.

        Parameters:
            - X (np.ndarray): input data
        """
        self.data = X
    
    def step(self):
        """
        Computes K-means clustering on the data
        """
        self.iteration += 1

        # compute distances
        distances = self._distances(self.mu)

        # assign labels
        labels = distances.argmin(axis=1)

        # update means
        new_mu = np.zeros_like(self.mu)
        for i in range(self.k):
            if np.any(labels == i):
                # normal update
                new_mu[i] = self.data[labels == i].mean(axis=0)
            else:
                # if empty cluster: random re‑init in domain
                new_mu[i] = np.random.rand(self.dim) * (self.domain[1] - self.domain[0]) + self.domain[0]

        self.mu = new_mu

        distances = self._distances(self.mu)
        self.labels = distances.argmin(axis=1)
    
    def plot(self, plotting = True):
        """
        Plots the K-means clustering results.

        Inputs:
            - plotting (bool): whether to plot the results
        """
        fig, ax = plt.subplots()

        # scatter data points
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, s=5, alpha=0.5)

        # scatter centroids
        ax.scatter(self.mu[:,0], self.mu[:,1], c = 'red', s=100, alpha=0.5)

        ax.set_title(f"Iteration {self.iteration}")
        image_path = os.path.join(self.images_dir, f"k_means_{self.iteration}.png")
        fig.savefig(image_path)
        if plotting:
            plt.show()
        plt.close(fig)
