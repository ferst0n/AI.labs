import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle


def regress():
    # Входной файл, содержащий данные
    input_file = 'data_singlevar_regr.txt'
    # Загрузка данных
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    # Разбивка данных на обучающий и тестовый наборы
    num_training = int(0.99 * len(X))
    num_test = len(X) - num_training
    # Тренировочные данные
    X_train, y_train = X[:num_training], y[:num_training]
    # Тестовые данные
    X_test, y_test = X[num_training:], y[num_training:]
    # Создание объекта линейного регрессора
    regressor = linear_model.LinearRegression()
    # Обучение модели с использованием обучающего набора
    regressor.fit(X_train, y_train)
    # Прогнозирование результата
    y_test_pred = regressor.predict(X_test)
    # Построение графика
    plt.scatter(X_test, y_test, color='green')
    plt.plot(X_test, y_test_pred, color='black', linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    # Вычисление метрических характеристик
    print("Linear regressor performance:")
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
    # Файл для сохранения модели
    output_model_file = 'model.pkl'
    # Сохранение модели
    with open(output_model_file, 'wb') as f:
        pickle.dump(regressor, f)

    # Загрузка модели
    with open(output_model_file, 'rb') as f:
        regressor_model = pickle.load(f)
    # Получение прогноза на тестовом наборе данных
    y_test_pred_new = regressor_model.predict(X_test)
    print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))


def multidimensionalRegress():
    # Входной файл, содержащий данные
    input_file = 'data_multivar_regr.txt'
    # Загрузка данных
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    # Разбивка данных на обучающий и тестовый наборы
    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training
    # Тренировочные данные
    X_train, y_train = X[:num_training], y[:num_training]
    # Тестовые данные
    X_test, y_test = X[num_training:], y[num_training:]

    # Создание модели линейного регрессора
    linear_regressor = linear_model.LinearRegression()
    # Обучение модели с использованием обучающих наборов
    linear_regressor.fit(X_train, y_train)

    # Прогнозирование результата
    y_test_pred = linear_regressor.predict(X_test)

    # Измерение метрических характеристик
    print("Linear Regressor performance:")
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
    print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

    # Полиномиальная регрессия
    polynomial = PolynomialFeatures(degree=10)
    X_train_transformed = polynomial.fit_transform(X_train)
    datapoint = [[7.75, 6.35, 5.56]]
    poly_datapoint = polynomial.fit_transform(datapoint)

    poly_linear_model = linear_model.LinearRegression()
    poly_linear_model.fit(X_train_transformed, y_train)
    print("\nLinear regression:\n", linear_regressor.predict(datapoint))
    print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

def costEstimate():
    data = datasets.load_boston()
    # Перемешивание данных
    X, y = shuffle(data.data, data.target, random_state=7)
    # Разбивка данных на обучающий и тестовый наборы
    num_training = int(0.7 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]
    # Создание регрессионной модели на основе SVM
    sv_regressor = SVR(kernel='linear', C=1, epsilon=0.1)
    # Обучение регрессора SVМ
    sv_regressor.fit(X_train, y_train)
    # Оценка эффективности работы регрессора
    y_test_pred = sv_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)
    print("\n#### Оценка эффективности ####")
    print("Среднеквадратическая ошибка =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))
    # Тестирование регрессора на тестовой точке данных
    test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
    print("\nPredicted price:", sv_regressor.predict([test_data])[0])

