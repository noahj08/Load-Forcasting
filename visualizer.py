import matplotlib.pyplot as plt 
def scatter(orig, pred, title, xlabel, ylabel, filename, show=True):
    plt.scatter(orig,pred)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    if show:
        plt.show()
