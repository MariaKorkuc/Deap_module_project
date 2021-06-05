import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def MLPParametersFeatures(numberFeatures,icls):
    activation = ["identity", "logistic", "tanh", "relu"]
    solver = ["lbfgs", "sgd", "adam"]
    learning_rate = ["constant", "invscaling", "adaptive"]
    genome = [
        activation[random.randint(0, len(activation) - 1)],
        solver[random.randint(0, len(solver) - 1)],
        random.uniform(0.0001, 1),  # alpha
        learning_rate[random.randint(0, len(learning_rate) - 1)],
        random.uniform(0.0001, 0.1),  # learn_rate_init
        True  # shuffle
    ]
    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)

def MLPParametersFeatureFitness(y,df,numberOfAtributtes,individual):
    split=5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop=[] #lista cech do usuniecia
    for i in range(numberOfAtributtes,len(individual)):
        if individual[i]==0: #gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i-numberOfAtributtes)
    dfSelectedFeatures=df.drop(df.columns[listColumnsToDrop], axis=1,inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = MLPClassifier(activation=individual[0], solver=individual[1], alpha=individual[2], learning_rate=individual[3], learning_rate_init=individual[4], shuffle=True, max_iter=10,  random_state=1)
    # mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4,
    #                     random_state=1, learning_rate_init=.1)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,

def mutationMLP(individual):
    numberParamer= random.randint(0,len(individual)-1)
    activation = ["identity", "logistic", "tanh", "relu"]
    solver = ["lbfgs", "sgd", "adam"]
    learning_rate = ["constant", "invscaling", "adaptive"]
    genome = [
        activation[random.randint(0, len(activation) - 1)],
        solver[random.randint(0, len(solver) - 1)],
        random.uniform(0.0001, 1),  # alpha
        learning_rate[random.randint(0, len(learning_rate) - 1)],
        random.uniform(0.0001, 0.1),  # learn_rate_init
        True  # shuffle
    ]
    if numberParamer < len(genome) - 1:
        individual[numberParamer] = genome[numberParamer]
    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0

def mutationMLP(individual):
    numberParamer= random.randint(0,len(individual)-1)
    activation = ["identity", "logistic", "tanh", "relu"]
    solver = ["lbfgs", "sgd", "adam"]
    learning_rate = ["constant", "invscaling", "adaptive"]
    genome = [
        activation[random.randint(0, len(activation) - 1)],
        solver[random.randint(0, len(solver) - 1)],
        random.uniform(0.0001, 1),  # alpha
        learning_rate[random.randint(0, len(learning_rate) - 1)],
        random.uniform(0.0001, 0.1),  # learn_rate_init
        True  # shuffle
    ]
    if numberParamer==0:
        individual[0]=genome[0]
    elif numberParamer==1:
        individual[1] = genome[1]
    elif numberParamer == 2:
        individual[2] = genome[2]
    elif numberParamer == 3:
        individual[3] = genome[3]
    elif numberParamer == 4:
        individual[4] = genome[4]
    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0