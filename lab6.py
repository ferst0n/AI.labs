from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import *
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle


def validityMeasures():
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type
    # Загрузка входных данных
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    # Разбиение входных данных на три класса на основании меток
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])
    # Визуализация входных данных
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='s')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='^')
    plt.title('Input data')
    # Разбивка данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=1)
    # Классификатор на основе ансамблевого обучения
    params = {'n_estimators': 60, 'max_depth': 3, 'random_state': 5}
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)

    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Training dataset')
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Test dataset')

    # Проверка работы классификатора
    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#" * 40 + "\n")
    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")

    # Вычисление параметров доверительности
    test_datapoints = np.array(
        [[1.2, 8], [25, 3], [2.8, 1], [9.4, 2.3], [4.5, 10], [15, 3], [2.3, 1.3], [4.5, 5], [7.8, 1], [4, 2]])
    print("\Измерение доверительности:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Класс-' + str(np.argmax(probabilities))
        print('\nТочка данных:', datapoint)
        print('Вероятности:', probabilities)
        print('Спрогнозированный класс:', predicted_class)
    # Визуализация точек данных
    visualize_classifier(classifier, test_datapoints, [0] * len(test_datapoints), 'Тестовые точки данных')
    plt.show()

def handleClassDisbalance():
    # Загрузка входных данных
    input_file = 'data_imbalance.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    # Разделение входных данных на два класса на основании меток
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    # Визуализация входных данных
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
                edgecolors='black', linewidth=1, marker='x')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o')
    plt.title('Входные данные')
    # Разбиение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    # Классификатор на основе предельно случайных лесов
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if len(sys.argv) > 1:
        if sys.argv[1] == 'balance':
            params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'class_weight':
                'balanced'}
        else:
            raise TypeError("Неправильный входной аргумент; должен быть 'balance'")
    classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Обучающий набор данных')
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Тестовый набор данных')
    # Вычисление показателей эффективности классификатора
    class_names = ['Class-0', 'Class-1']
    print("\n" + "#" * 40)
    print("\Эффективность классификатора на тренировочном наборе данных\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#" * 40 + "\n")
    print("#" * 40)
    print("\nЭффективность классификатора на тестовом наборе данных\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")
    plt.show()

def findOptimalTrainingParams():
    # Загрузка входных данных
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    # Разбиение данных на три класса на основании меток
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])
    # Разбиение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    # Определение сетки значений параметров
    parameter_grid = [{'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                      {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}]
    metrics = ['precision_weighted', 'recall_weighted']
    for metric in metrics:
        print("\n##### Поиск оптимальных значений параметров для", metric)
        print("\nВ таблице ниже в каждом столбце первым указано значение")
        print("n_estimators, вторым - max_depth, третьим - mean_test_score.")
        classifier = GridSearchCV(ExtraTreesClassifier(random_state=0), parameter_grid, cv = 5, scoring = metric)
        classifier.fit(X_train, y_train)

    print("\nGrid scores (оценка) для значений параметров:")
    par1 = classifier.cv_results_['param_n_estimators']
    par2 = classifier.cv_results_['param_max_depth']
    par3 = classifier.cv_results_['mean_test_score']
    par = [par1, par2, par3]
    for par in par:
        print(par)
    print("\nЛучшие значения параметров:", classifier.best_params_)
    y_pred = classifier.predict(X_test)
    print("\nОтчет о качестве классификатора:\n")
    print(classification_report(y_test, y_pred))


def calculatingRelativeImportanceFeatures():
    # Загрузка данных с ценами на недвижимость
    housing_data = datasets.load_boston()
    # Перемешивание данных
    X, y = shuffle(housing_data.data, housing_data.target, random_state=2)
    # Разбиение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=2)
    # Модель на основе регрессора AdaBoost
    regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=2)
    regressor.fit(X_train, y_train)
    # Вычисление показателей эффективности регрессора AdaBoost
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print("\nРЕГРЕССОР ADABOOST")
    print("Среднеквадратическая ошибка =", round(mse, 2))
    print("Объяснённая дисперсия =", round(evs, 2))
    # Извлечение важности признаков
    feature_importances = regressor.feature_importances_
    feature_names = housing_data.feature_names
    # Нормализация значений важности признаков
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # Сортировка и перестановка значений
    index_sorted = np.flipud(np.argsort(feature_importances))
    # Расстановка меток вдоль оси Х
    pos = np.arange(index_sorted.shape[0]) + 0.5
    # Построение столбчатой диаграммы
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Относительная важность')
    plt.title('Относительная важность, определённая посредством регрессора AdaBoost')
    plt.show()


