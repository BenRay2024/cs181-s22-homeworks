# CS 181, Spring 2022
# Homework 4

from re import S
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import cdist
from seaborn import heatmap

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
        self.cluster_sizes = []

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
            self.cluster_sizes.append(num_images_k)
            
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
    
    def get_cluster_sizes(self):
        return self.cluster_sizes
    
    def get_assignments(self):
        standardized_assignments = []
        for i, row in enumerate(self.responsibility_matrix):
            index = np.where(row == 1)
            standardized_assignments.append(index[0][0])
        return standardized_assignments


class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
            
    def fit(self, X):
        self.num_images, self.dim_images = X.shape # 300, 784
        self.X = X # copy of data
        self.cluster_sizes = []
        self.assignments = np.arange(self.num_images)
        distance_matrix = cdist(X, X)

        while (len(np.unique(self.assignments)) > 10):
            if self.linkage == "min":
                distance_matrix[distance_matrix == 0] = np.inf
                key = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
                distance_matrix[key[0], key[1]] = np.inf
                distance_matrix[key[1], key[0]] = np.inf
            elif self.linkage == "max":
                distance_matrix[distance_matrix == 0] = np.NINF
                key = np.unravel_index(distance_matrix.argmax(), distance_matrix.shape)
                distance_matrix[key[0], key[1]] = np.NINF
                distance_matrix[key[1], key[0]] = np.NINF
            else: # centroid
                distance_matrix[distance_matrix == 0] = np.inf
                key = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
            
            # Update assignments array
            for i in range(len(self.assignments)):
                min_key = min([key[0], key[1]])
                if self.assignments[i] == key[0] or self.assignments[i] == key[1]:
                    self.assignments[i] = min_key
            
            if self.linkage == "centroid":
                indexes = []
                for i, elt in enumerate(self.assignments):
                    if elt == min([key[0], key[1]]):
                        indexes.append(i)

                new_cluster_sum = np.zeros(self.dim_images)
                for i in indexes:
                    new_cluster_sum += self.X[i]
                
                centroid = new_cluster_sum / len(indexes)
                
                for i in indexes:
                    X[i] = centroid

                distance_matrix = cdist(X, X)

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        print(self.assignments)
        self.assignments_copy = copy.deepcopy(self.assignments)
        cluster_means = []
        for _ in range(n_clusters):
            min_n = min(self.assignments)
            indexes = []
            for i in range(len(self.assignments)):
                if self.assignments[i] == min_n:
                    indexes.append(i)
                    self.assignments[i] = 100000000 # big int
            len_cluster = len(indexes)
            self.cluster_sizes.append(len_cluster)
            cluster_mean = np.zeros(self.dim_images)
            for i in indexes:
                cluster_mean += (1/len_cluster) * self.X[i]
            cluster_means.append(cluster_mean)
        return np.array(cluster_means)
    
    def get_cluster_sizes(self):
        return self.cluster_sizes
    
    def get_assignments(self):
        counter = 0
        standardized_assignments = np.zeros(self.num_images)
        while(counter < 10):
            assignment_tracker = []
            min_cluster = min(self.assignments_copy)
            for i in range(len(standardized_assignments)):
                if self.assignments_copy[i] == min_cluster:
                    assignment_tracker.append(i)
                    self.assignments_copy[i] = 1000000000 # big int
            for i in assignment_tracker:
                standardized_assignments[i] = counter
            counter += 1
        return standardized_assignments
            

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
    return KMeansClassifier.get_cluster_sizes(), KMeansClassifier.get_assignments()

# ~~ Part 2 ~~
# _ = make_mean_image_plot(large_dataset, False)

# # ~~ Part 3 ~~
# TODO: Change this line! standardize large_dataset and store the result in large_dataset_standardized
mean = np.mean(large_dataset, axis=0)
sdev = np.std(large_dataset, axis=0)
for i in range(len(sdev)):
    if sdev[i] == 0:
        sdev[i] = 1

large_dataset_standardized = (large_dataset - mean) / sdev
# _ = make_mean_image_plot(large_dataset_standardized, True)

# K-Means with small dataset
kmeans_clusters, kmeans_assignments = make_mean_image_plot(small_dataset, False)

# Plotting code for part 4
def hac_run():
    LINKAGES = [ 'max', 'min', 'centroid' ]
    n_clusters = 10
    cluster_counts = []
    assignments = []
    fig = plt.figure(figsize=(10,10))
    plt.suptitle("HAC mean images with max, min, and centroid linkages")
    for l_idx, l in enumerate(LINKAGES):
        # Fit HAC
        hac = HAC(l)
        hac.fit(small_dataset)
        mean_images = hac.get_mean_images(n_clusters)
        cluster_counts.append(hac.get_cluster_sizes())
        assignments.append(hac.get_assignments())
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
    
    return cluster_counts, assignments

hac_clusters, hac_assignments = hac_run()

# # TODO: Write plotting code for part 5

plt.bar(np.arange(10), kmeans_clusters, width=0.4, label="K-Means")
plt.bar(np.arange(10), hac_clusters[0], width=0.4, label="HAC (max)")
plt.bar(np.arange(10), hac_clusters[1], width=0.4, label="HAC (min)")
plt.bar(np.arange(10), hac_clusters[2], width=0.4, label="HAC (centroid)")
plt.legend(loc="upper right")
plt.xlabel("Cluster index")
plt.ylabel("Number of images in cluster")
plt.show()

# # TODO: Write plotting code for part 6
print(kmeans_assignments)
print(hac_assignments)

map = np.zeros((10,10))
for i in range(len(kmeans_assignments)):
    x = int(kmeans_assignments[i])
    y = int(hac_assignments[0][i])
    map[x][y] += 1
heatmap(map)
plt.xlabel("K-Means")
plt.ylabel("HAC max")
plt.show()

# kmeans and HAC min
map = np.zeros((10,10))
for i in range(len(kmeans_assignments)):
    x = int(kmeans_assignments[i])
    y = int(hac_assignments[1][i])
    map[x][y] += 1
heatmap(map)
plt.xlabel("K-Means")
plt.ylabel("HAC min")
plt.show()

# kmeans and HAC centroid
map = np.zeros((10,10))
for i in range(len(kmeans_assignments)):
    x = int(kmeans_assignments[i])
    y = int(hac_assignments[2][i])
    map[x][y] += 1
heatmap(map)
plt.xlabel("K-Means")
plt.ylabel("HAC centroid")
plt.show()

# HAC max and HAC min
map = np.zeros((10,10))
for i in range(len(kmeans_assignments)):
    x = int(hac_assignments[0][i])
    y = int(hac_assignments[1][i])
    map[x][y] += 1
heatmap(map)
plt.xlabel("HAC max")
plt.ylabel("HAC min")
plt.show()

# HAC max and HAC centroid
map = np.zeros((10,10))
for i in range(len(kmeans_assignments)):
    x = int(hac_assignments[0][i])
    y = int(hac_assignments[2][i])
    map[x][y] += 1
heatmap(map)
plt.xlabel("HAC max")
plt.ylabel("HAC centroid")
plt.show()

# HAC min and HAC centroid
map = np.zeros((10,10))
for i in range(len(kmeans_assignments)):
    x = int(hac_assignments[1][i])
    y = int(hac_assignments[2][i])
    map[x][y] += 1
heatmap(map)
plt.xlabel("HAC min")
plt.ylabel("HAC centroid")
plt.show()
