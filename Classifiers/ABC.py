import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier as AB


def ABCFeatures(numberFeatures, icls):
    genome = list()

    listAlgorithm = ["SAMME", "SAMME.R"]
    learning_rate = random.uniform(0.01,1.5)
    n_estimators = random.randint(20,150)

    genome.append(random.choice(listAlgorithm))
    genome.append(learning_rate)
    genome.append(n_estimators)

    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))

    return icls(genome)

def ABCParametersFeatureFitness(y,df,numberOfAtributtes,individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = AB(algorithm=individual[0],
                   learning_rate=individual[1],
                   n_estimators=individual[2],
                   random_state=101)

    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,
                                                  predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,

def mutationAB(individual):
    numberParamer = random.randint(0, len(individual) - 1)

    if numberParamer == 0:
        listAlgorithm = ["SAMME", "SAMME.R"]
        individual[0] = random.choice(listAlgorithm)
    elif numberParamer == 1:
        learning_rate = random.uniform(0.001, 5.0)
        individual[1] = learning_rate
    elif numberParamer == 2:
        n_estimators = random.randint(20, 150)
        individual[2] = n_estimators
    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0