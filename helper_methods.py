import random
from matplotlib import pyplot as plt, ticker


def generate_random_colors(size):
    get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
    c = get_colors(size)
    return c

def plot_numbers(data=[],labels=[],x_zoom=0,y_zoom=0,title=""):
    c = generate_random_colors(41)
    for i in range(len(data.tolist())):
        plt.text(data[i,0],data[i,1],labels[i],c=c[labels[i]])
    plt.margins(x_zoom, y_zoom)
    plt.title(title)
    plt.show()

def plot_numbers_k_means(data=[],labels=[],labels_k_mean=[],x_zoom=0,y_zoom=0,title=""):
    c = generate_random_colors(len(labels_k_mean)+1)
    for i in range(len(data.tolist())):
        plt.text(data[i,0],data[i,1],labels[i],c=c[labels_k_mean[i]])
    plt.margins(x_zoom, y_zoom)
    plt.title(title)
    plt.show()

def plot_dots_2d(data=[],labels=[],title=""):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    plt.axis('tight')
    plt.show()


def plot_vals(X,Y,title):
    plt.scatter(X,Y)
    plt.title(title)
    plt.legend()
    plt.show()