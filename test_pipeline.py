from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
import numpy as np

class PLSExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        self.lb = LabelBinarizer()
    def fit(self, X, y):
        self.pls.fit(X, self.lb.fit_transform(y))
        return self
    def transform(self, X):
        return self.pls.transform(X)

X_train = np.random.rand(10, 1550)
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
pipe = Pipeline([('pls', PLSExtractor(n_components=2)), ('svm', SVC(kernel='rbf'))])
pipe.fit(X_train, y_train)
print(pipe.predict(X_train))
