import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import MyLinearRegression
from LinearRegression import l1_regularization, l2_regularization
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    x, y = datasets.make_regression(n_samples=500,
                                    n_features=1,
                                    n_targets=1,
                                    noise=15)
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
    input_param_name = "Profit"
    output_param_name = "Population"
    n_samples, n_features = np.shape(x)

    print(
        f'data: {len(x)}; train_data: {len(x_train)}; test_data: {len(x_test)}'
    )

    plt.style.use('seaborn')
    plt.scatter(x_train, y_train, label='Train data', c="r", alpha=0.5)
    plt.scatter(x_test, y_test, label='Test data', c="b", alpha=0.5)
    plt.xlabel(input_param_name)
    plt.ylabel(output_param_name)
    plt.legend()
    plt.show()

    num_iterations = 1500
    regularization = l2_regularization(alpha=0.5)
    learning_rate_list = [3 * 1e-3, 1e-3, 3 * 1e-4, 1e-4, 3*1e-5, 1e-5]
    error_list = []
    for learning_rate in learning_rate_list:
        print('-------------------------------------')
        print(f'learning_rate: {learning_rate}')
        # 可自行设置模型参数，如正则化，梯度下降轮数学习率等
        model = MyLinearRegression(num_iterations=num_iterations, learning_rate=learning_rate, regularization=regularization, gradient=True)
        model.fit(x_train, y_train)
        print('开始时的损失：', model.training_errors[0])
        print('训练后的损失：', model.training_errors[num_iterations - 1])
        print('-------------------------------------')
        error_list.append(model.training_errors)

    for error, learning_rate in zip(error_list, learning_rate_list):
        plt.plot(range(num_iterations),
                 error,
                 label=f'learning rate: {learning_rate}')
    plt.title('Different learning rate')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

    y_pred = model.predict(x_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    mse = mean_squared_error(y_test, y_pred)
    print("My mean squared error: %s" % (mse))

    plt.scatter(x_train, y_train, label='Train data', c="r", alpha=0.5)
    plt.scatter(x_test, y_test, label='Test data', c="b", alpha=0.5)
    plt.scatter(x_test, y_pred, label='Prediction', c='k', alpha=0.5)
    plt.plot(x_test, y_pred, label='Pred_line', c='g')
    plt.xlabel(input_param_name)
    plt.ylabel(output_param_name)
    plt.legend()
    plt.show()

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, y_predictions)
    print("Sklearn's mean squared error: %s" % (mse))
    plt.scatter(x_train, y_train, label='Train data', c="r", alpha=0.5)
    plt.scatter(x_test, y_test, label='Test data', c="b", alpha=0.5)
    plt.scatter(x_test, y_predictions, label='Prediction', c='k', alpha=0.5)
    plt.plot(x_test, y_predictions, label='Pred_line', c='g')
    plt.xlabel(input_param_name)
    plt.ylabel(output_param_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
