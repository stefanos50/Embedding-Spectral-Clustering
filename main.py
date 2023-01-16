import data_preprocessing
import embedding
import Spectral_Clustering
import numpy as np


X,y = data_preprocessing.get_olivetti()
#X,y = data_preprocessing.get_mnist()
X = np.array(X)
#embedding.compare_algorithms(X,y,2)
X = embedding.run_TSNE(X,y=y,zoom_x=1500,zoom_y=1500)

Spectral_Clustering.run(X,n_clusters=82,k_means_test_num=50,type="Radius",radius=2.6,y=y,plot_eig_num=150)