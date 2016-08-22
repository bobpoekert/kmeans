import numpy as np
import kmeans
d = 75
k = 1000
data = np.random.random(size=(100000, d))
labels = np.random.randint(k, size=data.shape[0]).astype(np.int32)
labels2 = labels.copy()
print kmeans.kmeans(data, labels, 1, k, 0.000001), np.bincount(labels2)
