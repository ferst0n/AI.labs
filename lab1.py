from sklearn import datasets
from sklearn import preprocessing
import numpy as np

import main


def firstTask():
    housePrices = datasets.load_boston()
    regress = datasets.load_linnerud()
    breastCancer = datasets.load_breast_cancer()
    """
        Datasets data 1.4
    """
    print("Boston data: \n", housePrices.data)
    print("Regress data: \n", regress.data)
    print("Cancer data: \n", breastCancer.data)
    """
        Datasets target 1.6
    """
    print("Boston target: \n", housePrices.target)
    print("Regress target: \n", regress.target)
    print("Cancer target: \n", breastCancer.target)
    """
        independent variables 1.8
    """
    print("Boston variable: \n", housePrices.data)
    print("Regress variable: \n", regress.data)
    print("Cancer variable: \n", breastCancer.data)
    """
        feature names 1.10
    """
    print("Boston names: \n", housePrices.feature_names)
    print("Regress names: \n", regress.feature_names)
    print("Cancer names: \n", breastCancer.feature_names)
    """
        Description 1.13
    """
    print("Boston description: \n", housePrices.DESCR)
    print("Regress description: \n", regress.DESCR)
    print("Cancer description: \n", breastCancer.DESCR)
    """
        graphic data 1.15
    """
    digit = datasets.load_digits()
    print("Digit №9: \n", digit.images[main.myNumber])


def secondTask():
    """
        Example 2.2
    """
    inputData = np.array([
        [5.1, -2.9, 3.3],
        [-1.2, 7.8, -6.1],
        [3.9, 0.4, 2.1],
        [7.3, -9.9, -4.5]
    ])
    dataBinarized = preprocessing.Binarizer(threshold=2.1).transform(inputData)
    print("Binarized data: \n", dataBinarized)
    """
        My data 2.4
    """
    myData = np.array([
        [8.4, 5.9, -10.3, 13.5],
        [3.2, -5.4, -9.9, 4.2],
        [-7.8, 13.5, -214, 3.7],
        [26, 26.9, -300.5, 2.5]
    ])
    for i in range(3, 6):
        dataBinarized = preprocessing.Binarizer(threshold=i).transform(myData)
        print("Binarized data, when threshold =", i, ": \n", dataBinarized)
    """
        average value and standard deviation for example data 2.5
    """
    print("Before:")
    print("Average value: \n", inputData.mean(axis=0))
    print("Standard deviation: \n", inputData.std(axis=0))
    print("After:")
    dataScaled = preprocessing.scale(inputData)
    print("Average value: \n", dataScaled.mean(axis=0))
    print("Standard deviation: \n", dataScaled.std(axis=0))
    """
        average value and standard deviation for my data 2.7
    """
    print("Before:")
    print("Average value: \n", myData.mean(axis=0))
    print("Standard deviation: \n", myData.std(axis=0))
    print("After:")
    dataScaled = preprocessing.scale(myData)
    print("Average value: \n", dataScaled.mean(axis=0))
    print("Standard deviation: \n", dataScaled.std(axis=0))
    """
        MinMax 2.8
    """
    dataScalerMinMax = preprocessing.MinMaxScaler(feature_range=(0, 1))
    dataScaledMinMax = dataScalerMinMax.fit_transform(inputData)
    print("Scaled Data: \n", dataScaledMinMax)
    for i in range(1, 4):
        dataScalerMinMax = preprocessing.MinMaxScaler(feature_range=(i, i+1))
        dataScaledMinMax = dataScalerMinMax.fit_transform(inputData)
        print("Scaled Data in range(",i,",",i+1,")\n", dataScaledMinMax)
    """
       normalized example data 2.11
    """
    dataNormalizedL1 = preprocessing.normalize(inputData, norm="l1")
    dataNormalizedL2 = preprocessing.normalize(inputData, norm="l2")
    print("L1 normalize: \n", dataNormalizedL1)
    print("L2 normalize: \n", dataNormalizedL2)
    """
        normalized my data 2.12
    """
    dataNormalizedL1 = preprocessing.normalize(inputData, norm="l1")
    dataNormalizedL2 = preprocessing.normalize(inputData, norm="l2")
    print("L1 normalize: \n", dataNormalizedL1)
    print("L2 normalize: \n", dataNormalizedL2)


