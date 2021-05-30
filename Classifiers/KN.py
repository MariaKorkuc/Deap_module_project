import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KN

def KNeighborsFeature(numberFeatures, icls):
    genome = list()

    n_neighbors = random.randint(1, 10)
    listWeights = ["uniform", "distance"]
    listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
    leaf_size = random.randint(5, 50)
    p = random.randint(1, 2)
    # listMetric = ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"]
    listMetric = ["minkowski"]

    genome.append(n_neighbors)
    genome.append(random.choice(listWeights))
    genome.append(random.choice(listAlgorithm))
    genome.append(leaf_size)
    genome.append(p)
    genome.append(random.choice(listMetric))

    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))

    return icls(genome)

def KNeighborsFeatureFitness(y,df,numberOfAtributtes,individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = KN(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2], leaf_size=individual[3],
                    p=individual[4], metric=individual[5])

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
        n_neighbors = random.randint(1, 10)
        individual[0] = n_neighbors
    elif numberParamer == 1:
        listWeights = ["uniform", "distance"]
        individual[1] = random.choice(listWeights)
    elif numberParamer == 2:
        listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
        individual[2] = random.choice(listAlgorithm)
    elif numberParamer == 3:
        leaf_size = random.randint(5, 50)
        individual[3] = leaf_size
    elif numberParamer == 4:
        p = random.randint(1, 2)
        individual[4] = p
    elif numberParamer == 5:
        # listMetric = ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"]
        listMetric = ["minkowski"]
        individual[5] = random.choice(listMetric)
    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0