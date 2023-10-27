import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from sklearn import datasets, metrics
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AffinityPropagation
from itertools import cycle
import csv
from sklearn.cluster import MeanShift, estimate_bandwidth


def clusteringQualityAssessment():
    # Загрузка набора данных iris
    iris = datasets.load_iris()
    # Разбиение данных на обучающий и тестовый наборы (в пропорции 80/20)
    indices = StratifiedKFold(n_splits=5)
    # Используем первый набор
    train_index, test_index = next(iter(indices.split(iris.data, iris.target)))
    # Извлечение обучающих данных и меток
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    # Извлечение тестовых данных и меток
    X_test = iris.data[test_index]
    y_test = iris.target[test_index]
    # Извлечение количества классов
    num_classes = len(np.unique(y_train))
    # Создание GММ
    classifier = GaussianMixture(n_components=num_classes, covariance_type='full',
                                 init_params='kmeans', max_iter=20)
    # Инициализация средних GММ
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in range(num_classes)])
    # Обучение GММ-классификатора
    classifier.fit(X_train)
    # Вычерчивание границ
    plt.figure()
    colors = 'bgr'
    for i, color in enumerate(colors):
        # Извлечение собственных значений и собственных векторов
        eigenvalues, eigenvectors = np.linalg.eigh(classifier.covariances_[i][:2, :2])
    # Нормализация первого собственного вектора
    norm_vec = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
    # Извлечение угла наклона
    angle = np.arctan2(norm_vec[1], norm_vec[0])
    angle = 180 * angle / np.pi
    # Масштабный множитель для увеличения эллипсов
    # (выбрано произвольное значение, которое нас удовлетворяет)
    scaling_factor = 8
    eigenvalues *= scaling_factor
    # Вычерчивание эллипсов
    ellipse = patches.Ellipse(classifier.means_[i, :2],
                              eigenvalues[0], eigenvalues[1], 180 + angle,
                              color=color)
    axis_handle = plt.subplot(1, 1, 1)
    ellipse.set_clip_box(axis_handle.bbox)
    ellipse.set_alpha(0.6)
    axis_handle.add_artist(ellipse)

    # Откладывание входных и тестовых данных на графике
    colors = 'bgr'
    for i, color in enumerate(colors):
        cur_data = iris.data[iris.target == i]
        plt.scatter(cur_data[:, 0], cur_data[:, 1], marker='o',
                    facecolors='none', edgecolors='black', s=40,
                    label=iris.target_names[i])
        test_data = X_test[y_test == i]
        plt.scatter(test_data[:, 0], test_data[:, 1], marker='s',
                    facecolors='black', edgecolors='black', s=40,
                    label=iris.target_names[i])

    # Вычисление прогнозных результатов для обучающих и тестовых данных
    y_train_pred = classifier.predict(X_train)
    accuracy_training = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print('Правильность (accuracy) для обучающих данных =', accuracy_training)

    y_test_pred = classifier.predict(X_test)
    accuracy_testing = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print('Правильность (accuracy) для тестовых данных =', accuracy_testing)
    plt.title('GMM-классификатор')
    plt.xticks(())
    plt.yticks(())
    plt.show()


def simpleSimilarityPropagationModel():
    # Формирование выборки
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=140, centers=centers, cluster_std=0.3,
                                random_state=0)

    # Вычисление распространения сходства
    af = AffinityPropagation(preference=-20).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print('Оценка числа кластеров: %d' % n_clusters_)
    print("Однородность: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Полнота: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels,
                                               average_method='arithmetic'))
    print("Силуэтный коэффициент: %0.3f"
          % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

    # Визуализация результатов
    plt.close('all')
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    plt.title('Оценочное число кластеров: %d' % n_clusters_)
    plt.show()

def segmentation():
    # Загрузка данных из входного файла
    input_file = 'sales.csv'
    file_reader = csv.reader(open(input_file, 'r'), delimiter = ',')
    X = []
    for count, row in enumerate(file_reader):
        if not count:
            names = row[1:]
            continue
        X.append([float(X) for X in row[1:]])
    # Преобразование данных в массив numpy
    X = np.array(X)
    # Оценка ширины окна входных данных
    bandwidth = estimate_bandwidth(X, quantile=0.9, n_samples=len(X))
    # Вычисление кластеризации методом сдвига среднего
    meanshift_model = MeanShift(bandwidth=bandwidth,
                                bin_seeding=True)
    meanshift_model.fit(X)
    labels = meanshift_model.labels_
    cluster_centers = meanshift_model.cluster_centers_
    num_clusters = len(np.unique(labels))
    print("\nNumber of clusters in input data =", num_clusters)
    print("\nCenters of clusters:")
    print('\t'.join([name[:3] for name in names]))
    for cluster_center in cluster_centers:
        print('\t'.join([str(int(X)) for X in cluster_center]))
    # Извлечение двух признаков в целях визуализации
    cluster_centers_2d = cluster_centers[:, 3:6]
    # Отображение центров кластеров
    plt.figure()
    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
                s=120, edgecolors= "black", facecolors = "none")
    offset = 0.25
    plt.xlim(cluster_centers_2d[:, 0].min() - offset *
        cluster_centers_2d[:, 0].ptp(),
        cluster_centers_2d[:, 0].max() + offset *
        cluster_centers_2d[:, 0].ptp(),)
    plt.ylim(cluster_centers_2d[:, 1].min() - offset *
             cluster_centers_2d[:, 1].ptp(),
             cluster_centers_2d[:, 1].max() + offset *
             cluster_centers_2d[:, 1].ptp())
    plt.title('Цeнтpы 2D-кластеров')
    plt.show()




