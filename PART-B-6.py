import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('income_clustering.csv')
print(df.head())

plt.scatter(df[['Age']],df[['Income($)']])
plt.xlabel('Age')
plt.ylabel('Annual Income($)')
plt.title("Age vs Income before Clustering")
plt.show()

kn = KMeans(n_clusters=3)
y_pred = kn.fit_predict(df[['Age','Income($)']])

df['clusters'] = y_pred

print(df.head())

df0 = df[df['clusters']==0]
df1 = df[df['clusters']==1]
df2 = df[df['clusters']==2]

plt.scatter(df0[['Age']],df0[['Income($)']],color='red',label='cluster 1')
plt.scatter(df1[['Age']],df1[['Income($)']],color='blue',label='cluster 2')
plt.scatter(df2[['Age']],df2[['Income($)']],color='green',label='cluster 3')
plt.scatter(kn.cluster_centers_[:,0],kn.cluster_centers_[:,1],marker='+',color='black',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title("K-means Clustering of Income and Age Data")
plt.legend()
plt.show()

sse = []
k_rng = range(1,10)
for k in k_rng:
    kn = KMeans(n_clusters = k)
    kn.fit(df[['Age','Income($)']])
    sse.append(kn.inertia_)

plt.plot(k_rng, sse)
plt.xlabel('Clusters')
plt.ylabel('Sum of Squared errors')
plt.title("Elbow Method")
plt.show()
