from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class KNN:
    def __init__(
        self, n_neighbors=[3, 5, 8, 15], leaf_size=[15, 30, 45], cv=5
    ):
        """
        KNN Class.

        This class initializes a KNN object for performing k-nearest neighbors classification.

        Args:
            n_neighbors (list): A list of integers representing the number of neighbors to consider.
                                Defaults to [3, 5, 8, 15].
            leaf_size (list): A list of integers representing the leaf size of the KD-tree.
                              Defaults to [15, 30, 45].
            cv (int): The number of cross-validation folds. Defaults to 5.

        """
        knn = KNeighborsClassifier()
        self.classifier = GridSearchCV(
            knn, {"n_neighbors": n_neighbors, "leaf_size": leaf_size}, cv=cv
        )

    def fit(self, train_embeddings, train_target):
        """
        Fit the KNN model using the provided training embeddings and target values.

        This method fits the KNN model using the provided training embeddings and target values.

        Args:
            train_embeddings: The training embeddings.
            train_target: The target values for training.

        """
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        """
        Make predictions using the trained KNN model on the provided test embeddings.

        This method makes predictions using the trained KNN model on the provided test embeddings.

        Args:
            test_embeddings: The test embeddings to predict on.

        Returns:
            The predicted target values.
        """
        return self.classifier.predict(test_embeddings)
