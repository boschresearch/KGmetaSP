from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class SVM:
    def __init__(self, C=[0.01, 0.1, 1, 10, 100], cv=5, random_state=42):
        """
        SVM Class.

        This class initializes an SVM object for performing Support Vector Machine classification.

        Args:
            C (list): A list of floats representing the regularization parameter C.
                      Defaults to [0.01, 0.1, 1, 10, 100].
            cv (int): The number of cross-validation folds. Defaults to 5.
            random_state (int): Random state, providing determinism.  Defaults to 42.
        """
        svc = SVC(random_state=42)
        self.classifier = GridSearchCV(svc, {"C": C}, cv=cv)

    def fit(self, train_embeddings, train_target):
        """
        Fit the SVM model using the provided training embeddings and target values.

        This method fits the SVM model using the provided training embeddings and target values.

        Args:
            train_embeddings: The training embeddings.
            train_target: The target values for training.

        """
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        """
        Make predictions using the trained SVM model on the provided test embeddings.

        This method makes predictions using the trained SVM model on the provided test embeddings.

        Args:
            test_embeddings: The test embeddings to predict on.

        Returns:
            The predicted target values.
        """
        return self.classifier.predict(test_embeddings)
