import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier as GP


def GaussianProcessFeature(numberFeatures, icls):
    genome = list()
    n_restarts_optimizer = random.randint(1, 10)
    max_iter_predict = random.randint(20, 100)
    multi_class_list = ["one_vs_rest", "one_vs_one"]

    genome.append(n_restarts_optimizer)
    genome.append(max_iter_predict)
    genome.append(random.choice(multi_class_list))

    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)

def GaussianProcessFeatureFitness(y,df,numberOfAtributtes,individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = GP(n_restarts_optimizer=individual[0], max_iter_predict=individual[1], multi_class=individual[2])

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

def mutationGaussianProcess(individual):
    numberParamer = random.randint(0, len(individual) - 1)

    if numberParamer == 0:
        n_restarts_optimizer = random.randint(1, 10)
        individual[0] = n_restarts_optimizer
    elif numberParamer == 1:
        max_iter_predict = random.randint(20, 100)
        individual[0] = max_iter_predict
    elif numberParamer == 2:
        multi_class_list = ["one_vs_rest", "one_vs_one"]
        individual[2] = random.choice(multi_class_list)
    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0