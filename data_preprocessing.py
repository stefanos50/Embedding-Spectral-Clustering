import numpy as np
import random
from matplotlib import pyplot as plt

def plot_images(data,size):
    data = np.array(data)
    amount = len(data)
    fig = plt.figure(figsize=(12, 12))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.reshape(data[random.randint(0, amount-1)],(size,size)))
    plt.show()

def get_olivetti():
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces(shuffle=True)
    X = faces.data
    y = faces.target
    #plot_images(X,64)
    return X,y

def get_mnist():
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    #plot_images(X,28)
    return X/255,y