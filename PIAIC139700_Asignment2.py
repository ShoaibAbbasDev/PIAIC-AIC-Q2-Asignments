#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt 


# In[16]:


np.random.seed(42)


# In[17]:


def enclideen_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


# In[24]:


class KMeans:
    def __init__(self , k=5, max_iters=100,plot_step=False):
        self.k=k
        self.max_iters=max_iters
        self.plot_step=plot_step
        
        self.clusters=[[] for _ in range(self.k)]
        self.centroids=[]
        
    # we use unsupervised machine learning so their is no label and we not create fit method we create predict method
    
    def predict(self, x):
        self.x=x
        self.n_samples , self.n_features= x.shape
        # initialize centroid 
        random_sample_indx=np.random.choice(self.n_samples, self.k , replace=False)
        self.centroids=[self.x[idx] for idx in random_sample_indx]
        
        # optimization 
        for _ in range(self.max_iters):
            # update cluster 
            self.clusters=self._create_clusters(self.centroids)
            if self.plot_step:
                self.plot()
            
            # update centroid 
            centroids_old =self.centroids
            self.centroids=self._get_centroid(self.clusters)
            if self.plot_step:
                self.plot()
            
            # check if covereged
            if self._is_converged( centroids_old , self.centroids):
                break
            
            
            # return cluster label 
    
        return self._get_cluster_labels(self.clusters)
    def _get_cluster_labels(self ,clusters):
        label=np.empty(self.n_samples)
        for cluster_idx , cluster in enumerate(clusters):
            for sample_idx in cluster:
                label[sample_idx]= cluster_idx
        return label
        
    def _create_clusters(self,centroids):
        clusters=[[] for _ in range(self.k)]
        for idx , sample in enumerate(self.x):
            centroid_idx=self._closest_centroid(sample , centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    
    def _closest_centroid(self ,sample , centroids):
        distances=[enclideen_distance(sample ,point) for point  in centroids]
        closest_idx=np.argmin(distances)
        return closest_idx
    
    def _get_centroid(self,clusters):
        centroids=np.zeros((self.k , self.n_features))
        for cluster_idx , cluster in enumerate(clusters):
            cluster_mean=np.mean(self.x[cluster],axis=0)
            centroids[cluster_idx]=cluster_mean
        return centroids
                           
    def _is_converged(self,  centroids_old , centroids):
        distances=[enclideen_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances)==0
    
    def plot(self):
        fig , ax =plt.subplot(figsize=(12,8))
        for i , index in enumerate(self.clusters):
            point=self.x[index].T 
            ax.scatter(*point)
        for point in self.centroids:
            ax.scattertter(*point, marker="X", color="black", linewidth=2)
        plt.show()


# In[25]:


from sklearn.datasets import make_blobs

x , y=make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True , random_state=42) 


# In[27]:


print(x.shape)
clusters=len(np.unique(y))
print(clusters)
kmean = KMeans(k=clusters, max_iters=150, plot_step=False )
y_pred =kmean.predict(x)
y_pred


# In[ ]:





# In[ ]:





# In[ ]:




