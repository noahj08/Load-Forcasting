import matplotlib.pyplot as plt

#Algorithm-specific visualizers

def visualize_regression(output, Y_test):
    (predictions, r2_score, mse_loss, coefficients) = output
    scatter(Y_test, predictions, "Actual vs. Predicted Demand (kW)",\
            "Actual (kW)", "Predicted (kW)", "baseline.jpg")
    print(f"R2 Score = {r2_score}")
    print(f"Mean Squared Error Loss = {mse_loss}")
    print(f"Coefficients = {coefficients}")

####################################################################################
# HELPER FUNCTIONS

# Scatter feature number i from an X array (i.e. X_train or X_test)
def scatter_feature(X,i,title, xlabel, ylabel, filename, show=True):
    y = X[i]
    x = range(len(y))
    scatter(x, y, title, xlabel, ylabel, filename, show)

def scatter(x, y, title, xlabel, ylabel, filename, show=True):
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    if show:
        plt.show()

def plot_CDF(x, y, title, xlabel, ylabel, filename):
    plt.plot(x, y, '.-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
    plt.savefig(f'{filename}.jpg')

def load_CDF(title, xlabel, ylabel, filename):
    (x,y) = pickle.load(open(f"{filename}.pickle", 'rb+'))
    plot_CDF(x,y, title, xlabel, ylabel, filename)

def CDF(x, title, xlabel, ylabel, filebase):
    x = sorted(x)
    y = np.asarray(x)
    y = y/np.sum(y)
    y = list(np.cumsum(y))
    dec = 0
    for i in range(1, len(x)):
        i -= dec
        if x[i] == x[i-1]:
            x.pop(i)
            y.pop(i)
            dec += 1
    pickle.dump((x, y), open(f'{filename}.pickle', 'wb+'))
    plot_CDF(x,y, title, xlabel, ylabel, filebase)
