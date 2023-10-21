import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from utilities import visualize_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score

def labelEncoding():
    input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(input_labels)
    print("\nLabel mapping:")
    for i, item in enumerate(encoder.classes_):
        print(item, '-->', i)

    test_labels = ['black', 'red', 'yellow']
    encoded_values = encoder.transform(test_labels)
    print("\nLabels =", test_labels)
    print("Encoded values =", list(encoded_values))

    encoded_values = [3, 4, 2, 0]
    decoded_list = encoder.inverse_transform(encoded_values)
    print("\nEncoded values =", encoded_values)
    print("Decoded labels =", list(decoded_list))

    X = np.array([[15, 9], [1, 4], [23, 8], [3.1, 5.5], [2, 5], [1, 5], [14, 0.4], [3.9, 0.9], [7, 1.5],
                  [4, 3], [1, 3], [10, 4.9]])
    y = np.array([1, 1, 3, 1, 1, 0, 2, 0, 2, 0, 0, 0])
    classifier = linear_model.LogisticRegression(solver='liblinear', C=10)
    classifier.fit(X, y)
    visualize_classifier(classifier, X, y)

def bayeClassifier():

    # Входной файл, содержащий данные
    input_file = 'my_data.txt'

    # Загрузка данных из входного файла
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Создание наивного байесовского классификатора
    classifier = GaussianNB()

    # Тренировка классификатора
    classifier.fit(X, y)

    # Predict the values for training data
    y_pred = classifier.predict(X)

    # Вычисление правильности классификатора
    accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
    print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

    # Визуализация результатов работы классификатора
    visualize_classifier(classifier, X, y)

    # Разбивка данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=2)
    classifier_new = GaussianNB()
    classifier_new.fit(X_train, y_train)
    y_test_pred = classifier_new.predict(X_test)

    # Вычисление правильности классификатора
    accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
    print("Accuracy of the new classifier =", round(accuracy, 2), "%")
    # Визуализация работы классификатора
    visualize_classifier(classifier_new, X_test, y_test)

    num_folds = 2
    accuracy_values = cross_val_score(classifier,
                                      X, y, scoring='accuracy', cv=num_folds)
    print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
    precision_values = cross_val_score(classifier,
                                       X, y, scoring='precision_weighted', cv=num_folds)
    print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
    recall_values = cross_val_score(classifier,
                                    X, y, scoring='recall_weighted', cv=num_folds)
    print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
    f1_values = cross_val_score(classifier,
                                X, y, scoring='f1_weighted', cv=num_folds)
    print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")

