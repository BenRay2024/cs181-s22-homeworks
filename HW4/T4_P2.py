# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")
data = np.load("P2_Autograder_Data.npy")

# Set random seed
np.random.seed(2)

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.losses = []
        self.iter_count = 0

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        self.num_images, self.dim_images = X.shape # 5000, 784

        # Intialize responsibility vectors
        initial_responsibilities = []
        for _ in range(self.num_images):
            r_vec = np.zeros(self.K)
            rand_index = np.random.randint(self.K)
            r_vec[rand_index] = 1
            initial_responsibilities.append(r_vec)
        initial_responsibility_matrix = np.array(initial_responsibilities) # 5000xK: r_vecs are rows
        
        for _ in range(10):
            # Compute cluster means
            cluster_means = []
            for k in range(self.K):
                # Number of images in class k
                num_images_k = np.sum(initial_responsibility_matrix[:,k])
                
                # Compute individual cluster mean
                cluster_mean = np.zeros(self.dim_images)
                for n in range(self.num_images):
                    cluster_mean += (1 / num_images_k) * initial_responsibility_matrix[n][k] * X[n]
                cluster_means.append(cluster_mean)
            
            self.cluster_means_matrix = np.array(cluster_means) # Kx784: mean_vecs are rows

            # Find distances of each image to each mean and update responsibility matrix
            self.responsibility_matrix = np.zeros((self.num_images, self.K))
            self.distances = []
            for n, x_vec in enumerate(X):
                x_distances = []
                for k in range(self.K):
                    distance = np.square(np.linalg.norm(x_vec - self.cluster_means_matrix[k]))
                    x_distances.append(distance)
                key = np.argmin(x_distances)
                self.responsibility_matrix[n][key] = 1
                self.distances.append(x_distances)

            initial_responsibility_matrix = self.responsibility_matrix

            # Calculate loss for each iteration
            loss = 0
            for n in range(self.num_images):
                loss += self.responsibility_matrix[n] @ self.distances[n]
            self.losses.append(loss)
            self.iter_count += 1

    def plot_loss(self):
        plt.plot(np.arange(0, self.iter_count), self.losses)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Residual Sum of Squares")
        plt.show()

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.cluster_means_matrix

class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.cluster_sizes = []
    
    def fit(self, X):
        self.X = X
        self.assignments = np.arange(300)
        distance_matrix = cdist(X, X)
        
        if self.linkage == "centroid":
            while(len(self.X) > 10):
                distance_matrix[distance_matrix == 0] = np.inf
                key = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)

                cluster1 = self.X[key[0]]
                cluster2 = self.X[key[0]]

                self.X = np.delete(self.X, [key[0], key[1]], 0)

                centroid = (cluster1 + cluster2) / 2

                self.X = np.vstack((self.X, centroid))
                
                distance_matrix = cdist(self.X, self.X)
        
        else: # min or max
            while (len(np.unique(self.assignments)) > 10):
                if self.linkage == "min":
                    distance_matrix[distance_matrix == 0] = np.inf
                    key = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
                    distance_matrix[key[0], key[1]] = np.inf
                    distance_matrix[key[1], key[0]] = np.inf
                else: # max
                    distance_matrix[distance_matrix == 0] = np.NINF
                    key = np.unravel_index(distance_matrix.argmax(), distance_matrix.shape)
                    distance_matrix[key[0], key[1]] = np.NINF
                    distance_matrix[key[1], key[0]] = np.NINF

                for i in range(len(self.assignments)):
                    if key[1] < key[0]:
                        if self.assignments[i] == key[0]:
                            self.assignments[i] = key[1]
                    else:
                        if self.assignments[i] == key[1]:
                            self.assignments[i] = key[0]

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        if self.linkage == "centroid":
            return self.X
        else: # max or min
            cluster_means = []
            for _ in range(n_clusters):
                min_n = min(self.assignments)
                indexes = []
                for i in range(len(self.assignments)):
                    if self.assignments[i] == min_n:
                        indexes.append(i)
                        self.assignments[i] = 100000000 # big int
                len_cluster = len(indexes)
                cluster_mean = np.zeros(784)
                for i in indexes:
                    cluster_mean += (1/len_cluster) * self.X[i]
                cluster_means.append(cluster_mean)
            return np.array(cluster_means)
    
    def get_cluster_sizes(self):
        return self.cluster_sizes
            

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    KMeansClassifier.plot_loss()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.show()

# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)

# ~~ Part 3 ~~
# TODO: Change this line! standardize large_dataset and store the result in large_dataset_standardized
mean = np.mean(large_dataset, axis=0)
sdev = np.std(large_dataset, axis=0)
for i in range(len(sdev)):
    if sdev[i] == 0:
        sdev[i] == 1

large_dataset_standardized = (large_dataset - mean) / sdev
make_mean_image_plot(large_dataset_standardized, True)

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10

fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.show()

# # TODO: Write plotting code for part 5
cluster_sizes = hac.get_cluster_sizes()
plt.plot(np.arange(10), cluster_sizes)
plt.xlabel("Cluster index")
plt.ylabel("Number of images in cluster")
plt.show()

# # TODO: Write plotting code for part 6