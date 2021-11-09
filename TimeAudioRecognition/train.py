import pickle
import numpy as np

with open("data/x", "rb") as f:
    x = pickle.load(f)
with open("data/y", "rb") as f:
    y = pickle.load(f)

x = np.vstack(x)
print(x.shape)