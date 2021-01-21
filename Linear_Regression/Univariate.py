import matplotlib.pyplot as plt
from UnivariateLinearRegression import UnivariateLinearRegression
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

    num_iterations = 200
    learning_rate_list = [1e-2, 3 * 1e-3, 1e-3, 3 * 1e-4, 1e-4]
    cost_list = []
    for learning_rate in learning_rate_list:
        print('-------------------------------------')
        print(f'learning_rate: {learning_rate}')
        LR = UnivariateLinearRegression(data=x_train,
                                        labels=y_train,
                                        alpha=learning_rate,
                                        num_iterations=num_iterations)
        (theta, cost_history) = LR.train()
        print('开始时的损失：', cost_history[0])
        print('训练后的损失：', cost_history[-1])
        print('-------------------------------------')
        cost_list.append(cost_history)

    for cost, learning_rate in zip(cost_list, learning_rate_list):
        plt.plot(range(num_iterations),
                 cost,
                 label=f'learning rate: {learning_rate}')
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

    y_predictions = LR.predict(data=x_test, labels=y_test, theta=theta)
    print(f'自己模型训练后的损失: {mean_squared_error(y_test, y_predictions)}')
    plt.scatter(x_train, y_train, label='Train data', c="r", alpha=0.5)
    plt.scatter(x_test, y_test, label='Test data', c="b", alpha=0.5)
    plt.scatter(x_test, y_predictions, label='Prediction', c='k', alpha=0.5)
    plt.plot(x_test, y_predictions, label='Pred_line', c='g')
    plt.xlabel(input_param_name)
    plt.ylabel(output_param_name)
    plt.legend()
    plt.show()

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)
    print(f'Sklearn模型训练后的损失: {mean_squared_error(y_test, y_predictions)}')
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
