import helper_methods
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
import time

def run_TSNE(X=[],n_components=2,random_state=42,learning_rate='auto',init='pca',method="barnes_hut",n_iter_without_progress=300, perplexity=30,n_jobs=-1,early_exaggeration=12, n_iter=1000,angle=0.8,metric="euclidean",min_grad_norm=0.00000001,zoom_x=2000,zoom_y=2000,y=[]):
    start_time = time.time()
    X_transformed = TSNE(n_components=n_components ,random_state=random_state,n_iter_without_progress=n_iter_without_progress,method=method, learning_rate=learning_rate,init=init, perplexity=perplexity,n_jobs=n_jobs,early_exaggeration=early_exaggeration, n_iter=n_iter,angle=angle,metric=metric,min_grad_norm=min_grad_norm).fit_transform(X)
    print("tSNE execution time: " + str(time.time() - start_time))
    helper_methods.plot_numbers(X_transformed,y,zoom_x,zoom_y)
    helper_methods.plot_dots_2d(X_transformed,y)
    return X_transformed

def compare_algorithms(X,y,n_components):
    #TSNE
    start_time = time.time()
    X_transformed = TSNE(n_components=n_components).fit_transform(X)
    print("tSNE execution time: " + str(time.time() - start_time))
    helper_methods.plot_numbers(X_transformed,y,500,500)
    helper_methods.plot_dots_2d(X_transformed,y)

    #ISOMAP
    start_time = time.time()
    X_transformed = Isomap(n_components=n_components).fit_transform(X)
    print("ISOMAP execution time: " + str(time.time() - start_time))
    helper_methods.plot_numbers(X_transformed,y,500,500)
    helper_methods.plot_dots_2d(X_transformed,y)

    #LLE
    start_time = time.time()
    X_transformed = LocallyLinearEmbedding(n_components=n_components).fit_transform(X)
    print("LLE execution time: " + str(time.time() - start_time))
    helper_methods.plot_numbers(X_transformed,y,500,500)
    helper_methods.plot_dots_2d(X_transformed,y)

    #SPECTRAL EMBEDDING
    start_time = time.time()
    X_transformed = SpectralEmbedding(n_components=n_components).fit_transform(X)
    print("SPECTRAL EMBEDDING execution time: " + str(time.time() - start_time))
    helper_methods.plot_numbers(X_transformed,y,500,500)
    helper_methods.plot_dots_2d(X_transformed,y)













