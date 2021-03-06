import numpy as np
from initialize_with_zeros import initialize_with_zeros
from propagate import propagate
from optimize import optimize
from sigmoid import sigmoid
from predict import predict


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    # 用0初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 梯度下降
    parameters, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从参数表中取出 w 和 b
    w = parameters["w"]
    b = parameters["b"]

    # 预测测试集和训练集中的样本
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("训练集精度：{} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试集精度：{} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d
