import numpy as np


def fit_ridge_regression(X, Y, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: value of beta (1 dimensional np.array)
    """
    ll = np.empty(X.shape[1])
    ll.fill(l)
    ll[0] =0
    D = X.shape[1]  # dimension + 1
    beta = np.zeros(D)  # FIXME: ridge regression formula.
    beta = np.dot(np.dot( np.linalg.inv(np.dot(X.T, X) + ll * np.identity(D) ), X.T) ,Y)
    return beta


def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    XX, ll, mean, sd = FeatureScaling(X, l)
    beta = np.zeros(XX.shape[1])
    ll[0] =0
    for s in range(max_steps):
        loss1 = 0;
        loss2= loss(XX, Y, beta);
        ridge= normalized_gradient(XX,Y,beta,ll)
        if (np.absolute(loss2 - loss1) < epsilon):
            break
        loss1 = loss2
        beta = beta - step_size * ridge;

    #b1 = beta[0] - np.sum( np.divide( np.dot(mean, beta[1:beta.shape[0]]), sd))
    b2 = np.divide(beta[1: beta.shape[0]], sd)
    b1 = beta[0] - sum(b2 * mean)
    # b2 = beta[1] / sd
    return np.append(b1, b2)


def loss(X, Y, beta):
    """
    Calculate sum of error squares divided by number of points.

    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :return: 1/N * SUM (y - x beta)^2
    """
    loss_ =  np.sum (( Y - np.dot(X, beta))**2) / len(Y)
    return loss_

def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    bb=0
    for i in range(Y.shape[0]):
        bb = bb + np.dot((Y[i] - np.dot(X[i], beta)), X[i])
    return (-2*bb + np.dot(2*l, beta))/ X.shape[0]


def FeatureScaling(X, l):
    mean = np.zeros(X.shape[1] -1)
    sd =np.zeros(X.shape[1] - 1);
    temp = X.copy()
    lambda_mod = np.zeros(X.shape[1])
    lambda_mod[0] = l

    for i in range(1, X.shape[1]):
        mean[i-1] = np.mean(X.T[i])

    for j in range(1, X.shape[1]):
        for i in range(X.shape[0]):
            temp[i][j] = X[i][j] - mean[j-1]
            sd[j-1] = sd[j-1] + (X[i][j] - mean[j-1])**2
        sd[j-1] = np.sqrt(sd[j-1] / X.shape[0])
        lambda_mod[j] = l / sd[j-1]

    for j in range(1, X.shape[1]):
        for i in range(X.shape[0]):
            temp[i][j] = temp[i][j] / sd[j-1]
    return temp, lambda_mod**2, mean, sd

def stochastic_gradient_descent(X, Y, epsilon=0.0001, l=1, step_size=0.01,
                                max_steps=1000):
    """
    Implement gradient descent using stochastic approximation of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    beta1 = np.zeros(X.shape[1])
    beta2 = np.zeros(X.shape[1])

    index_array = np.arange(X.shape[0])
    np.random.shuffle(index_array)
    i = 0
    XX, ll, mean, sd = FeatureScaling(X, l)

    for s in range(max_steps):
        i = s % XX.shape[0]
        bb = np.dot((Y[index_array[i]] - np.dot(XX[index_array[i]], beta2)), XX[index_array[i]])
        #i = (i+1)  % XX.shape[0]

        ridge=  ((-2*bb)) + np.dot(2*ll, beta2) / X.shape[0]

        if ( (np.linalg.norm( beta2- beta1) ) < epsilon* np.linalg.norm(beta2) ):
            break

        beta1 = beta2
        beta2 = beta2 - step_size * ridge;
    print(s)
    b2 = np.divide(beta2[1: beta2.shape[0]], sd)
    b1 = beta2[0] - sum(b2 * mean)

    return np.append(b1,b2)
