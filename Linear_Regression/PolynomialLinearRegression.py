import numpy as np


class PolynomialLinearRegression:
    def __init__(self, data, labels, alpha, polynomial, num_iterations):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        self.polynomial = polynomial
        self.data = self.normalize(data)
        self.rate = np.max(labels) - np.min(labels)
        self.labels = labels / self.rate
        self.num_examples = self.data.shape[0]
        self.theta = np.zeros((1, self.num_examples))
        self.alpha = alpha
        self.num_iterations = num_iterations

    def normalize(self, data):
        data_process = []
        data_process.append([1. for _ in range(len(data))])
        for i in range(self.polynomial):
            for j in range(len(data[0])):
                ini_data = [data[k, j] for k in range(len(data))]
                temp = (ini_data - np.mean(ini_data)) / np.std(ini_data)
                data_process.append(list(np.power(temp, i + 1)))
        return np.array(data_process)

    @staticmethod
    def hypothesis(data, theta):
        predictions = np.dot(theta, data)
        return predictions

    def cost_function(self):
        """
        损失计算方法
        """
        delta = PolynomialLinearRegression.hypothesis(
            self.data, self.theta) - self.labels.T
        cost = (1 / 2) * np.dot(delta, delta.T) / self.num_examples
        return cost[0][0]

    def predict(self, data, labels, theta):
        """
        用训练的参数模型，与预测得到回归值结果
        """
        data_process = self.normalize(data)
        predictions = PolynomialLinearRegression.hypothesis(
            data_process, theta)
        return predictions[0] * self.rate

    def gradient_step(self):
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        """
        prediction = PolynomialLinearRegression.hypothesis(
            self.data, self.theta)
        delta = prediction - self.labels.T
        self.theta = self.theta - self.alpha * (
            1 / self.num_examples) * np.dot(delta, self.data.T)

    def gradient_descent(self):
        """
        实际迭代模块，会迭代num_iterations次
        """
        cost_history = []
        for _ in range(self.num_iterations):
            self.gradient_step()
            cost = self.cost_function()
            cost_history.append(cost)
        return cost_history

    def train(self):
        """
        训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent()
        return self.theta, cost_history
