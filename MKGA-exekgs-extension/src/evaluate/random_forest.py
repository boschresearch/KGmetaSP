from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForest:
    def __init__(
        self,
        random_state=42,
        n_estimators=[10, 20, 40],
        max_depth=[3, 5, 10, None],
        cv=10,
    ):
        """
        RandomForest Class.

        This class initializes a RandomForest object for performing random forest classification.

        Args:
            random_state (int): The random seed for reproducibility. Defaults to 42.
            n_estimators (list): A list of integers representing the number of trees in the forest.
                                 Defaults to [10, 20, 40].
            max_depth (list): A list of integers and None representing the maximum depth of the trees.
                              Defaults to [3, 5, 10, None].
            cv (int): The number of cross-validation folds. Defaults to 10.

        """
        gscv = RandomForestClassifier(random_state=random_state)
        self.classifier = GridSearchCV(
            gscv,
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
            },
            cv=cv,
        )

    def fit(self, train_embeddings, train_target):
        """
        Fit the RandomForest model using the provided training embeddings and target values.

        This method fits the RandomForest model using the provided training embeddings and target values.

        Args:
            train_embeddings: The training embeddings.
            train_target: The target values for training.

        """
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        """
        Make predictions using the trained RandomForest model on the provided test embeddings.

        This method makes predictions using the trained RandomForest model on the provided test embeddings.

        Args:
            test_embeddings: The test embeddings to predict on.

        Returns:
            The predicted target values.
        """
        return self.classifier.predict(test_embeddings)
