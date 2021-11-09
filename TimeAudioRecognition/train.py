import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

with open("data/x", "rb") as f:
    x = pickle.load(f)
with open("data/y", "rb") as f:
    y = pickle.load(f)

x = np.vstack(x)
y = np.array(y, dtype='int')
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

# clf = SVC(C=5)
clf = RandomForestClassifier(oob_score=True)
clf.fit(train_x, train_y)
print(sum(clf.predict(train_x) == train_y) / len(train_y))
print(sum(clf.predict(test_x) == test_y) / len(test_y))
