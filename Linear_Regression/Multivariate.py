import matplotlib.pyplot as plt
from MultivariateLinearRegression import MultivariateLinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly
import plotly.graph_objs as go


def main():
    x, y = datasets.make_regression(n_samples=500,
                                    n_features=2,
                                    n_targets=1,
                                    noise=15)
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
    input_param_name_1 = "Size_house"
    input_param_name_2 = "Num_bedroom"
    output_param_name = "Price"

    print(
        f'data: {len(x)}; train_data: {len(x_train)}; test_data: {len(x_test)}'
    )

    ax = plt.axes(projection='3d')
    ax.scatter3D(x_train[:, 0],
                 x_train[:, 1],
                 y_train,
                 label='Train data',
                 c='r',
                 alpha=0.5)
    ax.scatter3D(x_test[:, 0],
                 x_test[:, 1],
                 y_test,
                 label='Test data',
                 c='b',
                 alpha=0.5)
    ax.set_xlabel(input_param_name_1)
    ax.set_ylabel(input_param_name_2)
    ax.set_zlabel(output_param_name)
    plt.legend()
    plt.show()

    # Configure the plot with training dataset.
    # plot_training_trace = go.Scatter3d(
    #     x=x_train[:, 0].flatten(),
    #     y=x_train[:, 1].flatten(),
    #     z=y_train.flatten(),
    #     name='Training Set',
    #     mode='markers',
    #     marker={
    #         'size': 10,
    #         'opacity': 1,
    #         'line': {
    #             'color': 'rgb(255, 255, 255)',
    #             'width': 1
    #         },
    #     }
    # )
    # plot_test_trace = go.Scatter3d(
    #     x=x_test[:, 0].flatten(),
    #     y=x_test[:, 1].flatten(),
    #     z=y_test.flatten(),
    #     name='Test Set',
    #     mode='markers',
    #     marker={
    #         'size': 10,
    #         'opacity': 1,
    #         'line': {
    #             'color': 'rgb(255, 255, 255)',
    #             'width': 1
    #         },
    #     }
    # )
    # plot_layout = go.Layout(
    #     title='Date Sets',
    #     scene={
    #         'xaxis': {'title': input_param_name_1},
    #         'yaxis': {'title': input_param_name_2},
    #         'zaxis': {'title': output_param_name}
    #     },
    #     margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    # )
    # plot_data = [plot_training_trace, plot_test_trace]
    # plot_figure = go.Figure(data=plot_data, layout=plot_layout)
    # plotly.offline.plot(plot_figure)

    num_iterations = 200
    learning_rate_list = [1e-2, 3 * 1e-3, 1e-3, 3 * 1e-4, 1e-4]
    cost_list = []
    for learning_rate in learning_rate_list:
        print('-------------------------------------')
        print(f'learning_rate: {learning_rate}')
        LR = MultivariateLinearRegression(data=x_train,
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
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_train[:, 0],
                 x_train[:, 1],
                 y_train,
                 label='Train data',
                 c='r',
                 alpha=0.5)
    ax.scatter3D(x_test[:, 0],
                 x_test[:, 1],
                 y_test,
                 label='Test data',
                 c='b',
                 alpha=0.5)
    ax.scatter3D(x_test[:, 0],
                 x_test[:, 1],
                 y_predictions,
                 label='Prediction',
                 c='k',
                 alpha=0.5)
    ax.set_xlabel(input_param_name_1)
    ax.set_ylabel(input_param_name_2)
    ax.set_zlabel(output_param_name)
    plt.legend()
    plt.show()

    plt.scatter(x_test[:, 0],
                y_predictions,
                label='Size_house-Price',
                c='k',
                alpha=0.5)
    plt.scatter(x_test[:, 1],
                y_predictions,
                label='Num_bedroom-Price',
                c='g',
                alpha=0.5)
    plt.ylabel('feature')
    plt.ylabel(output_param_name)
    plt.legend()
    plt.show()

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)
    print(f'Sklearn模型训练后的损失: {mean_squared_error(y_test, y_predictions)}')
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_train[:, 0],
                 x_train[:, 1],
                 y_train,
                 label='Train data',
                 c='r',
                 alpha=0.5)
    ax.scatter3D(x_test[:, 0],
                 x_test[:, 1],
                 y_test,
                 label='Test data',
                 c='b',
                 alpha=0.5)
    ax.scatter3D(x_test[:, 0],
                 x_test[:, 1],
                 y_predictions,
                 label='Prediction',
                 c='k',
                 alpha=0.5)
    ax.set_xlabel(input_param_name_1)
    ax.set_ylabel(input_param_name_2)
    ax.set_zlabel(output_param_name)
    plt.legend()
    plt.show()

    plt.scatter(x_test[:, 0],
                y_predictions,
                label='Size_house-Price',
                c='k',
                alpha=0.5)
    plt.scatter(x_test[:, 1],
                y_predictions,
                label='Num_bedroom-Price',
                c='g',
                alpha=0.5)
    plt.ylabel('feature')
    plt.ylabel(output_param_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
