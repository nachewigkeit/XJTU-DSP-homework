import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

with open("data/x", "rb") as f:
    x = pickle.load(f)
with open("data/y", "rb") as f:
    y = pickle.load(f)

x = np.vstack(x)
y = np.array(y, dtype='int')
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
print(train_x.shape)

# clf = SVC(C=5)
clf = RandomForestClassifier(oob_score=True)
clf.fit(train_x, train_y)
yPred = clf.predict(test_x)
print(f1_score(test_y, yPred, average='macro'))
matrix = confusion_matrix(test_y, yPred)
plt.xticks(range(10))
plt.yticks(range(10))
plt.imshow(matrix, cmap='Blues')
plt.savefig(r"image/confusion.png", bbox_inches='tight')
