import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import check_grad
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("./fashion-mnist_test.csv")
data.head()
data = data[data.label.isin([5,6,7,8])]
data.label -= 5
X, Y = np.array(data.iloc[:, 1:]), np.array(data.label)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
n_classes = len(set(y_train))
n_features = X_test.shape[1]


def  softmax(Z, epsilon=1e-9):

    exp = np.exp(Z-np.max(Z, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return epsilon + (exp / norms)

def infer(W, X):

    X = np.hstack([X, np.ones([np.shape(X)[0], 1])])
    
    return np.dot(X, W)

eta = 1e-2

def one_hot_encode(labels_list, max_number):

    samples_number = len(labels_list)
    b = np.zeros((samples_number, max_number))
    b[np.arange(samples_number), labels_list] = 1
    return b

def loss(W, X, Y):

    sm = softmax(infer(W, X))
    L2 = np.dot(np.transpose(W), W) * eta / 2
    H = np.array([-np.log(np.sum(Y[i] * sm[i]) ) for i in range(np.shape(Y)[0])])
    return (np.sum((H)) / len(H)) + np.sum(L2)
    

def get_grad(W, X, Y):

    T = softmax(infer(W, X))

    X = np.hstack([X, np.ones([np.shape(X)[0], 1])])
    dE = np.transpose(X) * (T - Y) + W * eta
    #dE = np.dot(np.transpose(X) , (T - Y)) / np.shape(X)[0] + W * eta
    
    return dE

def draw_dataset(draw_mean = False):
    f, *axes = plt.subplots(2, 5, sharey=True, figsize=(10, 4))
    for label in range(10):
        if draw_mean:
            pic = data[data.label == label].iloc[:,1:].values.mean(axis=0).reshape((28,28))
        else:
            pic = data[data.label == label].iloc[0,1:].values.reshape((28,28))
        axis_cur = axes[0][label//5][label%5] 
        axis_cur.imshow(pic)
    plt.show()

def train(X_train, y_train, batch_size=50, num_epoch=1, n_classes=n_classes, step=1e-3, plot_loss=True):

    losses = []
    
    n_features = X_train.shape[1]
    
    # Initialize from normal distribution
    w = np.random.randn(n_features+1, n_classes)/n_features
    
    # perform gradient descent
    
    for epoch in range(num_epoch):
        for iter_num, (x_batch, y_batch) in enumerate(zip(np.split(X_train, batch_size), np.split(y_train, batch_size))):
            y_batch_tmp = one_hot_encode(y_batch, n_classes)

            for i in range(np.shape(x_batch)[0]):
                x_batch_tmp = np.reshape(x_batch[i], (1, np.shape(x_batch[iter_num])[0]))
                w = w - step * get_grad(w, x_batch_tmp, np.reshape(y_batch_tmp[i], (1, np.shape(y_batch_tmp[i])[0]) ))
               
            losses.append(loss(w, x_batch, one_hot_encode(y_batch, n_classes)))

    # draw learning curve 
    if plot_loss:
        plt.plot(losses)
        plt.title("Loss")
        plt.xlabel("epochs")
        #plt.show()
        
    return w

def make_prediction(X, W):
    """
        Take X with shape [n_samples, n_features]
        return: np.array of labels with shape [n_samples]
    """
    probability_matrix = infer(W, X)
    return np.array([np.argmax(t) for t in probability_matrix])

def main():

    """ X = np.array([[0.1, 0.5], [1.1, 2.3], [-1.1, -2.3], [-1.5, -2.5]])
    W = np.array([[0.1,0.2,0.3], [0.1,0.2,0.3], [0.01, 0.1, 0.1]])
    Y = np.array([[1,0,0],[0,1,0], [0,0,1], [0,0,1]])
    res = softmax(infer(W, X))
    print(res)
    print(loss(W,X,Y))
    num_features = 4
    num_classes = 3
    num_points = 1
    w, X, y = np.random.random((num_features+1, num_classes)),\
          np.random.random((num_points, num_features)),\
          (np.random.randint(0, num_classes, num_points))

    y_onehot = one_hot_encode(y, num_classes)
    func = lambda x: loss(x.reshape(w.shape), X, y_onehot)

    grad = lambda x: get_grad(x.reshape(w.shape), X, y_onehot).flatten()

    print('error = %s' % check_grad(func, grad, w.flatten()))
    """    
    
    print("number of classes: ", n_classes)
    print("number of features: ", n_features)
    print("train length: ", len(X_train))
    print("test length: ", len(X_test))

    W = train(X_train, y_train, num_epoch = 3)
    y_pred = make_prediction(X_test, W)
    
    print(classification_report(y_test, y_pred))
    plt.show()
    

if __name__ == "__main__":
    main()