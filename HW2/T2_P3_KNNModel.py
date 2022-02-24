import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None
    
    def __distance(self, star1, star2):
        mag1 = star1[0]
        mag2 = star2[0]
        temp1 = star1[1]
        temp2 = star2[1]
        return (((mag1 - mag2) / 3) ** 2) + ((temp1 - temp2) ** 2)
    
    def __takeSecond(self, elem):
        _, knn_reg = elem
        return knn_reg

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        y_hat = np.zeros(((X_pred.shape)[0], 1))
        
        # List of lists: will store the distances between every x_test point and x_0 to x_6
        k_dist = []

        for vec in X_pred:
            lst = []
            for i, comparison in enumerate(X_pred):
                distance = self.__distance(vec, comparison)
                lst.append((i, distance))
            lst.sort(key=self.__takeSecond, reverse=True)
            k_dist.append(lst)

        for n, lst in enumerate(k_dist):
            knn_reg = []
            for i, (j, _dist) in enumerate(lst):
                if i < self.K:
                    knn_reg.append(self.y[j])
                else:
                    y_hat[n] = knn_reg
                    break
        
        for i, preds in enumerate(y_hat):
            y_hat[i] = max(set(preds), key = preds.count)

        return y_hat

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y