#
# Giacomo De Pietro 2025
#

import csv
import numpy as np
import sys
from PyFastBDT import FastBDT


class IrisData:
    def __init__(self, filename, n_features=4, delimiter=','):
        self._X = []
        self._labels = []
        self._weights = []

        rows = []
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if len(row) < n_features + 1:
                    raise ValueError(f"Malformed line: {row}")
                features = [float(x) for x in row[:n_features]]
                label = int(row[n_features])
                rows.append(features)
                self._labels.append(label)
                self._weights.append(1.0)

        self._X = [[row[i] for row in rows] for i in range(n_features)]

    def getX(self):
        return self._X

    def getY(self, target_label):
        return [label == target_label for label in self._labels]

    def getW(self):
        return self._weights


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_file>")
        sys.exit(1)

    filename = sys.argv[1]

    # Load data
    data = IrisData(filename)
    X = np.array(data.getX()).T
    y = data.getY(1)
    w = data.getW()

    # Instantiate FastBDT Classifier
    classifier = FastBDT.Classifier(
        binning=np.array([5, 5, 5, 5], dtype=np.uint32),
        nTrees=10,
        depth=3,
        shrinkage=0.1,
        subsample=0.5,
        transform2probability=True,
        purityTransformation=np.array([False, False, False, False], dtype=bool),
        sPlot=False,
        flatnessLoss=-1,
        numberOfFlatnessFeatures=0
    )

    classifier.fit(X, y, w)

    def get_iris_score(classifier, X, y):
        y = np.array(y, dtype=np.float32)
        preds = classifier.predict(X)
        diff = y - preds
        return float(np.sum(diff ** 2))

    print("Score", get_iris_score(classifier, X, y))

    # Save classifier to weightfile
    weightfile = "unittest_py.weightfile"
    classifier.save(weightfile)

    # Load classifier back from weightfile
    classifier2 = FastBDT.Classifier()
    classifier2.load(weightfile)
    print("Score", get_iris_score(classifier2, X, y))
