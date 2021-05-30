import multiprocessing
from deap import base
from deap import creator
from deap import tools
import random
import time
import pandas as pd
from deapWorkspace.Classifiers import DTC, SVC, RFC, ABC

# SVC, DTC, RFC, ABC
curr_clf_type = 'ABC'

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
        'SVC': (SVC.SVCParametersFeatures, SVC.SVCParametersFeatureFitness, SVC.mutationSVC),
        'DTC': (DTC.DTFeatures, DTC.DTParametersFeatureFitness, DTC.mutationDT),
        'RFC': (RFC.RFCFeatures, RFC.RFCParametersFeatureFitness, RFC.mutationRF),
        'ABC': (ABC.ABCFeatures, ABC.ABCParametersFeatureFitness, ABC.mutationAB),
    }[clf_type]

def individual(icls, start=-10, stop=10):
    genome = list()
    genome.append(random.uniform(start,stop))
    genome.append(random.uniform(start,stop))

    return icls(genome)

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

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        #
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]
        print(f"{curr_clf_type}: Best individual is {best_ind}, {best_ind.fitness.values}")
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
