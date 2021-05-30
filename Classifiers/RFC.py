import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RF


def RFCFeatures(numberFeatures, icls):
    genome = list()
    listCriterion = ["gini", "entropy"]
    listMaxFeatures = ["auto", "sqrt", "log2", None]

    max_depth = random.randint(1,50)

    if random.randint(1,2)%2:
        min_samples_split = random.uniform(0.1, 1.0)
    else:
        min_samples_split = random.randint(2, 5)

    if random.randint(1,2)%2:
        min_samples_leaf = random.uniform(0.01, 0.5)
    else:
        min_samples_leaf = random.randint(1, 4)

    n_estimators = random.randint(50,150)

    genome.append(random.choice(listCriterion))
    genome.append(random.choice(listMaxFeatures))
    genome.append(max_depth)
    genome.append(min_samples_split)
    genome.append(min_samples_leaf)
    genome.append(n_estimators)

    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))

    return icls(genome)

def RFCParametersFeatureFitness(y,df,numberOfAtributtes,individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = RF(criterion=individual[0],
                   max_features=individual[1],
                   max_depth=individual[2],
                   min_samples_split=individual[3],
                   min_samples_leaf=individual[4],
                   n_estimators=individual[5],
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

def mutationRF(individual):
    numberParamer = random.randint(0, len(individual) - 1)

    if numberParamer == 0:
        listCriterion = ["gini", "entropy"]
        individual[0] = random.choice(listCriterion)
    elif numberParamer == 1:
        listMaxFeatures = ["auto", "sqrt", "log2", None]
        individual[1] = random.choice(listMaxFeatures)
    elif numberParamer == 2:
        max_depth = random.randint(1, 50)
        individual[2] = max_depth
    elif numberParamer == 3:
        if random.randint(1, 2) % 2:
            min_samples_split = random.uniform(0.1, 1.0)
        else:
            min_samples_split = random.randint(2, 20)
        individual[3] = min_samples_split
    elif numberParamer == 4:
        if random.randint(1, 2) % 2:
            min_samples_leaf = random.uniform(0.01, 0.5)
        else:
            min_samples_leaf = random.randint(1, 5)
        individual[4] = min_samples_leaf
    elif numberParamer == 5:
        n_estimators = random.randint(50, 200)
        individual[5] = n_estimators
    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0