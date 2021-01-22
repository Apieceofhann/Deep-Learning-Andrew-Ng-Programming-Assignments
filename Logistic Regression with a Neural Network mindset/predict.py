import numpy as np
from sigmoid import sigmoid


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计算A向量
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        temp = A
        temp[np.where(A > 0.5)] = 1
        temp[np.where(A <= 0.5)] = 0
        Y_prediction = temp
    assert (Y_prediction.shape == (1, m))
    return Y_prediction
