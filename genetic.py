import multiprocessing
from deap import base
from deap import creator
from deap import tools
import random
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier as DTC

# SVC, DTC,
curr_clf_type = 'DTC'

selmethod = 'best'
crossover = 'onepoint'
minimalization = False

sizePopulation = 100
probabilityMutation = 0.5
probabilityCrossover = 0.8
numberIteration = 100

mean_val=5
std=10
indpb=0.06

pd.set_option('display.max_columns', None)
df=pd.read_csv("data.csv",sep=',')
y=df['Status']
df.drop('Status',axis=1,inplace=True)
df.drop('ID',axis=1,inplace=True)
df.drop('Recording',axis=1,inplace=True)
numberOfAtributtes= len(df.columns)

# mms = MinMaxScaler()
# df_norm = mms.fit_transform(df)
# clf = DTC()
# scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)

def choose_clf(clf_type):
    return {
        'SVC': (SVCParametersFeatures, SVCParametersFeatureFitness, mutationSVC),
        'DTC': (DTFeatures, DTParametersFeatureFitness, mutationDT),
    }[clf_type]

def individual(icls, start=-10, stop=10):
    genome = list()
    genome.append(random.uniform(start,stop))
    genome.append(random.uniform(start,stop))

    return icls(genome)

def SVCParametersFeatures(numberFeatures,icls):
    genome = list()
    # kernel
    listKernel = ["linear","rbf", "poly", "sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])
    #c
    k = random.uniform(0.1, 100)
    genome.append(k)
    #degree
    genome.append(random.uniform(0.1,5))
    #gamma
    gamma = random.uniform(0.001,2)
    genome.append(gamma)
    # coeff
    coeff = random.uniform(0.01, 4)
    genome.append(coeff)
    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)
def SVCParametersFeatureFitness(y,df,numberOfAtributtes,individual):
    split=5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop=[] #lista cech do usuniecia
    for i in range(numberOfAtributtes,len(individual)):
        if individual[i]==0: #gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i-numberOfAtributtes)
    dfSelectedFeatures=df.drop(df.columns[listColumnsToDrop], axis=1,inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = SVC(kernel=individual[0],C=individual[1],degree=individual[2],
        gamma=individual[3],coef0=individual[4],random_state=101)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,
def mutationSVC(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer==0:
        # kernel
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0]=listKernel[random.randint(0, 3)]
    elif numberParamer==1:
        #C
        k = random.uniform(0.1,100)
        individual[1]=k
    elif numberParamer == 2:
        #degree
        individual[2]=random.uniform(0.1, 5)
    elif numberParamer == 3:
        #gamma
        gamma = random.uniform(0.01, 1)
        individual[3]=gamma
    elif numberParamer ==4:
        # coeff
        coeff = random.uniform(0.1, 1)
        individual[2] = coeff
    else: #genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0

def DTFeatures(numberFeatures, icls):
    genome = list()
    listCriterion = ["gini", "entropy"]
    listSplitter = ["best", "random"]
    listMaxFeatures = ["auto", "sqrt", "log2", None]

    max_depth = random.randint(1,50)

    if random.randint(1,2)%2:
        min_samples_split = random.uniform(0.1, 1.0)
    else:
        min_samples_split = random.randint(2, 20)

    if random.randint(1,2)%2:
        min_samples_leaf = random.uniform(0.01, 0.5)
    else:
        min_samples_leaf = random.randint(1, 5)

    genome.append(random.choice(listCriterion))
    genome.append(random.choice(listSplitter))
    genome.append(random.choice(listMaxFeatures))
    genome.append(max_depth)
    genome.append(min_samples_split)
    genome.append(min_samples_leaf)

    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))

    return icls(genome)
def DTParametersFeatureFitness(y,df,numberOfAtributtes,individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = DTC(criterion=individual[0], splitter=individual[1], max_features=individual[2],max_depth=individual[3],
                    min_samples_split=individual[4], min_samples_leaf=individual[5], random_state=101)

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
def mutationDT(individual):
    numberParamer = random.randint(0, len(individual) - 1)

    if numberParamer == 0:
        listCriterion = ["gini", "entropy"]
        individual[0] = random.choice(listCriterion)
    elif numberParamer == 1:
        listSplitter = ["best", "random"]
        individual[1] = random.choice(listSplitter)
    elif numberParamer == 2:
        listMaxFeatures = ["auto", "sqrt", "log2", None]
        individual[2] = random.choice(listMaxFeatures)
    elif numberParamer == 3:
        max_depth = random.randint(1, 50)
        individual[3] = max_depth
    elif numberParamer == 4:
        if random.randint(1, 2) % 2:
            min_samples_split = random.uniform(0.1, 1.0)
        else:
            min_samples_split = random.randint(2, 20)
        individual[4] = min_samples_split
    elif numberParamer == 5:
        if random.randint(1, 2) % 2:
            min_samples_leaf = random.uniform(0.01, 0.5)
        else:
            min_samples_leaf = random.randint(1, 5)
        individual[5] = min_samples_leaf
    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0

def register_toolbox(clf_type=curr_clf_type):
    # choosing the fitness function
    if minimalization:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    else:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)


    # toolbox prepration
    toolbox = base.Toolbox()

    parameters = choose_clf(clf_type)

    features = parameters[0]
    fitness = parameters[1]
    mutation_type = parameters[2]


    toolbox.register('individual',features, numberOfAtributtes, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness,y,df,numberOfAtributtes)

    # choosing the selection method
    if selmethod == 'tournament':
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif selmethod == 'best':
        toolbox.register("select", tools.selBest)
    elif selmethod == 'random':
        toolbox.register("select", tools.selRandom)
    elif selmethod == 'worst':
        toolbox.register("select", tools.selWorst)
    elif selmethod == 'roulette':
        toolbox.register("select", tools.selRoulette)
    elif selmethod == 'doubletournament':
        toolbox.register("select", tools.selDoubleTournament, tournsize=3, parsimony=2, fitness_first=False)
    else:
        toolbox.register("select", tools.selStochasticUniversalSampling)
    #
    # choosing crossover method
    if crossover == 'onepoint':
        toolbox.register("mate", tools.cxOnePoint)
    elif crossover == 'uniform':
        toolbox.register("mate", tools.cxUniform)
    elif crossover == 'twopoint':
        toolbox.register("mate", tools.cxTwoPoint)
    else:
        toolbox.register("mate", tools.cxOrdered)

    toolbox.register("mutate", mutation_type)
    return toolbox

def loop(g):
    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + listElitism
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        # length = len(pop)
        # mean = sum(fits) / length
        # sum2 = sum(x * x for x in fits)
        # std = abs(sum2 / length - mean ** 2) ** 0.5
        #
        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        # results.append([mean, std, best_ind.fitness.values])
        results.append(best_ind.fitness.values)
        #
    print("-- End of (successful) evolution --")

def toFile(results, t1, t2):
    with open('results.txt', 'w') as f:
        for result in results:
            f.write(str(result) + '\n')
        f.write(str(t2 - t1))

toolbox = register_toolbox()

pop = toolbox.population(n=sizePopulation)
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

g = 0
numberElitism = 1
results = []

t1 = time.time()
loop(g)
t2 = time.time()

toFile(results, t1, t2)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)
