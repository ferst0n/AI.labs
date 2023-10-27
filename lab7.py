import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle




def clusteringDataK():
    # Загрузка входных данных
    X = np.loadtxt('data_clustering.txt', delimiter=',')
    num_clusters = 5
    # Включение входных данных в график
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.title('Входные данные')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # Создание объекта КМеаns
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    # Обучение модели кластеризации КМеаns
    kmeans.fit(X)
    # Определение шага сетки
    step_size = 0.01
    # Определение сетки точек для отображения границ
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    # Прогнозирование выходных меток для всех точек сетки
    output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.clf()
    plt.imshow(output, interpolation='nearest',
               extent=(x_vals.min(), x_vals.max(),
                       y_vals.min(), y_vals.max()),
               cmap=plt.cm.Paired,
               aspect='auto',
               origin='lower')
    # Отображение входных точек
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
                edgecolors='black', s=80)
    # Отображение центров кластеров
    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker='o', s=210, linewidths=4, color='black',
                zorder=12, facecolors='black')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.title('Границы кластеров')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def middleShift():
    # Загрузка данных из входного файла
    X = np.loadtxt('data_clustering.txt', delimiter=', ')
    # Оценка ширины окна для Х
    bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
    # Кластеризация данных методом сдвига среднего
    meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
    meanshift_model.fit(X)
    # Извлечение центров кластеров
    cluster_centers = meanshift_model.cluster_centers_
    print('\nЦентры кластеров:\n', cluster_centers)
    # Оценка количества кластеров
    labels = meanshift_model.labels_
    num_clusters = len(np.unique(labels))
    print("\nКоличество кластеров во входных данных =", num_clusters)
    # Отображение на графике точек и центров кластеров
    plt.figure()
    markers = 'o*xvs'
    for i, marker in zip(range(num_clusters), markers):
        # Отображение на графике точек, принадлежащих текущему кластеру
        plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black')

    # Отображение на графике центра кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black',
             markersize=15)
    plt.title('Кластеры')
    plt.show()

def clusteringQualityAssessment():
    # Загрузка данных из входного файла
    X = np.loadtxt('my_data_quality.txt', delimiter=',')
    # Включение входных данных в график
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.title('Входные данные')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # Инициализация переменных
    scores = []
    values = np.arange(70, 90)
    # Итерирование в определённом диапазоне значений
    for num_clusters in values:
        # Обучение модели кластеризации КМеаns
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
        kmeans.fit(X)
        # Получить силуэтную оценку
        score = metrics.silhouette_score(X, kmeans.labels_,
                                         metric='euclidean', sample_size=len(X))
    print("\nКоличество кластеров =", num_clusters)
    print("Силуэтная оценка =", score)

    scores.append(score)
    # Отображение силуэтных оценок на графике
    plt.figure()
    plt.bar(values, scores, width=0.7, color='black', align='center')
    plt.title('Силуэтная оценка числа кластеров')
    # Извлечение наилучшей оценки и оптимального количества кластеров
    num_clusters = np.argmax(scores) + values[0]
    print('\nОптимальное количество кластеров =', num_clusters)
    plt.show()



