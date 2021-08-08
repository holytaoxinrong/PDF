from tree import PowerTreeClassifier
from sklearn.datasets import load_iris
import numpy as np


if __name__ == '__main__':
    dtc = PowerTreeClassifier()
    X, y = load_iris(return_X_y=True)
    dtc.fit([X], [y])
    result = dtc.predict(X)
    print(result)