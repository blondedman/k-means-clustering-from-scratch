from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

# plt.scatter(X[:,0], X[:,1], s = 20, linewidths=5)
# plt.show()

colors = 10 * ["g","r","b","k","c"]

class KMeans:
    
    def __init__(self, k=2, tol=0.001, iter=300):
        self.k = k
        self.tol = tol
        self.iter = iter
        
    def fit(self, data):
        
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i]
                    
        for i in range(self.iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
            
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[c]) for c in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
        
            prev = dict(self.centroids)
        
            for classification in self.centroids:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            
            optimized = True
            
            for c in self.centroids:
                orig = prev[c]
                curr = self.centroids[c]
                if np.sum((curr-orig)/orig*100) > self.tol:
                    optimized = False
            
            if optimized:
                break
            
    
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[c]) for c in self.centroids]
        classifications = distances.index(min(distances))
        return classifications
    
clf = KMeans()
clf.fit(X)

print(clf.centroids)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=20, linewidths=5)

print(clf.classifications)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=20, linewidths=5)

plt.show()