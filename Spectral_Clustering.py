import networkx as nx
from sklearn.neighbors import kneighbors_graph
import scipy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csgraph
import helper_methods

def show_graph_with_labels(adjacency_matrix,y=[]):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    labeldict = {}
    for i in range(0,len(y)):
        labeldict[i] = str(y[i])
    nx.draw(gr, node_size=100,with_labels=True,labels=labeldict)
    plt.show()

def compute_graph_laplacian(X=[], type="NN",metric="minkowski",p=2,mode="connectivity",radius=1,n_jobs=-1,n_neighbors=1,y=[]):
    if type == "NN":
        A = kneighbors_graph(X, n_neighbors=n_neighbors,metric=metric,p=p,mode=mode,n_jobs=n_jobs)
        show_graph_with_labels(A.toarray(),y)
        A = (1 / 2) * (A + A.T)
    else:
        A = radius_neighbors_graph(X, radius=radius,metric=metric,p=p,mode=mode,n_jobs=n_jobs)
        show_graph_with_labels(A.toarray(), y)
        A = (1 / 2) * (A + A.T)
    A = A.toarray()
    L = csgraph.laplacian(A, normed=False)
    return L

def compute_eig(graph_laplacian):
    eigenvals, eigenvcts = scipy.linalg.eig(graph_laplacian)
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    return eigenvals, eigenvcts

def short_and_plot_eig(eigenvals, eigenvcts, num , plot_num):
    eigenvals_sorted_indices = np.argsort(eigenvals)
    indices = eigenvals_sorted_indices[: num]
    projection = eigenvcts[:, indices.squeeze()]
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]
    helper_methods.plot_vals(range(1, eigenvals_sorted_indices.size + 1),eigenvals_sorted,"Eigvalues")
    helper_methods.plot_vals(range(1, plot_num+1), eigenvals_sorted[0:plot_num], "Eigvalues")
    return projection

def run_k_means_algorithm(X, n_clusters):
    k_means = KMeans(n_clusters=n_clusters,max_iter=1000)
    k_means.fit(X)
    cluster = k_means.predict(X)
    return cluster

def k_means_test(X,num_k):
    inertias = []
    k_candidates = range(1, num_k)
    for k in k_candidates:
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        inertias.append(k_means.inertia_)
    helper_methods.plot_vals(k_candidates,inertias,"K Means Inertias")

def run(X=[],n_clusters=1,type="NN",metric="minkowski",p=2,mode="connectivity",radius=1,n_jobs=-1,n_neighbors=1,plot_eig_num=5,k_means_test_num=10,y=[],zoom_x=1000,zoom_y=1000):
    graph_laplacian = compute_graph_laplacian(X,type=type,metric=metric,p=p,mode=mode,radius=radius,n_jobs=n_jobs,n_neighbors=n_neighbors,y=y)
    eigenvals, eigenvcts = compute_eig(graph_laplacian)
    eig_vecs = short_and_plot_eig(eigenvals, eigenvcts, n_clusters,plot_eig_num)
    k_means_test(eig_vecs,k_means_test_num)
    cluster = run_k_means_algorithm(eig_vecs, n_clusters)
    helper_methods.plot_dots_2d(X,cluster,"Spectral Clustering")
    helper_methods.plot_numbers_k_means(X,y,cluster,zoom_x,zoom_y)


