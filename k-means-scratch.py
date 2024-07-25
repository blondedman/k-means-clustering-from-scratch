from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

# plt.scatter(X[:,0],X[:,1], s = 20, linewidths=5)
# plt.show()

colors = 10 * ["g","r","b","k","c"]

class KMeans:
    
    def __init__(self, k=2, tol=0.001, iter=300):
        self.k = k
        self.tol = tol
        self.iter = iter
    
    def fit(self, data):
        pass
    
    def predict(self,data):
        pass