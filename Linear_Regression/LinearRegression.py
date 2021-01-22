import numpy as np


# L1正则化
class l1_regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    # L1正则化的方差
    def __call__(self, w):
        loss = np.sum(np.fabs(w))
        return self.alpha * loss

    # L1正则化的梯度
    def grad(self, w):
        return self.alpha * np.sign(w)


# L2正则化
class l2_regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    # L2正则化的方差
    def __call__(self, w):
        loss = w.T.dot(w)
        return self.alpha * 0.5 * float(loss)

    # L2正则化的梯度
    def grad(self, w):
        return self.alpha * w


class MyLinearRegression:
    """
    Parameters:
    -----------
    regularization: l1_regularization or l2_regularization or None
        正则化
    gradient: Bool
        是否采用梯度下降法或正规方程法。
        若使用了正则化，暂只支持梯度下降
    num_iterations: int
        梯度下降的轮数
    learning_rate: float
        梯度下降学习率
    regularization: l1_regularization or l2_regularization or None
        正则化
    gradient: Bool
        是否采用梯度下降法或正规方程法。
        若使用了正则化，暂只支持梯度下降
    """

    def __init__(self, num_iterations=1000, learning_rate=1e-2, regularization=None, gradient=True):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        if regularization is None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    def initialize_weights(self, n_features):
        """
        初始化参数
        """
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, x, y):
        m_samples, n_features = x.shape
        self.initialize_weights(n_features)
        x = np.insert(x, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
        self.training_errors = []
        if self.gradient == True:
            # 梯度下降
            for i in range(self.num_iterations):
                y_pred = x.dot(self.w)
                delta = (y_pred - y)
                loss = np.mean(0.5 * delta ** 2) + self.regularization(self.w)  # 计算loss
                self.training_errors.append(loss)
                w_grad = x.T.dot(y_pred - y) + self.regularization.grad(self.w)  # (y_pred - y).T.dot(X)，计算梯度
                self.w = self.w - self.learning_rate * w_grad  # 更新权值w
        else:
            # 正规方程
            x = np.ndarray(x)
            y = np.ndarray(y)
            x_T_x = x.T.dot(x)
            x_T_x_I_x_T = x_T_x.I.dot(x.T)
            x_T_x_I_x_T_x_T_y = x_T_x_I_x_T.dot(y)
            self.w = x_T_x_I_x_T_x_T_y

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        y_pred = x.dot(self.w)
        return y_pred
