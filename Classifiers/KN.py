import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KN


def KNeighborsFeature(numberFeatures, icls):
    # genome = list()
    # n_neighbors = random.randint(1, 10)
    # leaf_size = random.randint(5, 50)
    # p = [1, random.uniform(1.001, 1.99), 2]
    # p = p[random.randint(0, 2)]
    #
    # genome.append(n_neighbors)
    # genome.append(leaf_size)
    # genome.append(p)

    p = [1, random.uniform(1.001, 1.99), 2]
    genome = [
        random.randint(1, 10),
        random.randint(10, 50),
        p[random.randint(0, 2)],
    ]

    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))

    return icls(genome)


#     listWeights = ["uniform", "distance"]
#     listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
#     # listMetric = ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"]
#     listMetric = ["minkowski"]
#     genome.append(random.choice(listWeights))
#     genome.append(random.choice(listAlgorithm))
#     genome.append(random.choice(listMetric))

def KNeighborsFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    # estimator = KN(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2], leaf_size=individual[3],
    #                 p=individual[4], metric=individual[5])

    estimator = KN(n_neighbors=individual[0], leaf_size=individual[1], p=individual[2])

    # estimator = KN()

    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,
                                                  predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split


def mutationKNeighbors(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        individual[0] == random.randint(1, 10)
    elif numberParamer == 1:
        individual[1] = random.randint(10, 50)
    elif numberParamer == 2:
        p = [1, random.uniform(1.001, 1.99), 2]
        individual[2] = p[random.randint(0, 2)]
    else:  # genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
#
#     if numberParamer == 0:
#         n_neighbors = random.randint(1, 10)
#         individual[0] = n_neighbors
#     elif numberParamer == 1:
#         listWeights = ["uniform", "distance"]
#         individual[1] = random.choice(listWeights)
#     elif numberParamer == 2:
#         listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
#         individual[2] = random.choice(listAlgorithm)
#     elif numberParamer == 3:
#         leaf_size = random.randint(5, 50)
#         individual[3] = leaf_size
#     elif numberParamer == 4:
#         p = random.randint(1, 2)
#         individual[4] = p
#     elif numberParamer == 5:
#         # listMetric = ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"]
#         listMetric = ["minkowski"]
#         individual[5] = random.choice(listMetric)
#     else:
#         if individual[numberParamer] == 0:
#             individual[numberParamer] = 1
#         else:
#             individual[numberParamer] = 0
