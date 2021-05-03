import multiprocessing

from deap import base
from deap import creator
from deap import tools
import random
import time

selmethod = 'tournament'
crossover = 'heuristic'
mutation = 'gaussian'
minimalization = True

sizePopulation = 100
probabilityMutation = 0.2
probabilityCrossover = 0.8
numberIteration = 100

mean_val=5
std=10
indpb=0.06


def individual(icls, start=-10, stop=10):
    genome = list()
    genome.append(random.uniform(start,stop))
    genome.append(random.uniform(start,stop))

    return icls(genome)

def fitnessFunction(individual):
    result = (individual[0] + 2* individual[1] - 7)**2 + (2*individual[0] + individual[1] -5)**2
    return result,

def heuristic(ind1, ind2):
    if (ind1[0] - ind1[0]) * (ind2[1] - ind2[1]) < 0:
        return ind1, ind2

    k1 = random.random()
    k2 = random.random()
    x1_1 = k1 * abs(ind2[0] - ind1[0]) + min(ind2[0], ind1[0])
    x2_1 = k1 * abs(ind2[1] - ind1[1]) + min(ind2[1], ind1[1])
    x1_2 = k2 * abs(ind2[0] - ind1[0]) + min(ind2[0], ind1[0])
    x2_2 = k2 * abs(ind2[1] - ind1[1]) + min(ind2[1], ind1[1])

    ind1[0] = x1_1
    ind1[1] = x2_1
    ind2[0] = x1_2
    ind2[1] = x2_2

    return ind1, ind2

# choosing the fitness function
if minimalization:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
else:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


# toolbox prepration
toolbox = base.Toolbox()
toolbox.register('individual', individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitnessFunction)

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

# TODO: arithmetic & heristic
# choosing crossover method
if crossover == 'onepoint':
    toolbox.register("mate", tools.cxOnePoint)
elif crossover == 'uniform':
    toolbox.register("mate", tools.cxUniform)
elif crossover == 'twopoint':
    toolbox.register("mate", tools.cxTwoPoint)
elif crossover == 'arithmetic':
    pass
elif crossover == 'heuristic':
    toolbox.register("mate", heuristic)
else:
    toolbox.register("mate", tools.cxOrdered)


# choosing mutation method
if mutation == 'gaussian':
    toolbox.register("mutate", tools.mutGaussian, mu=mean_val, sigma=std, indpb=indpb)
elif mutation == 'shuffle':
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indpb)
elif mutation == 'multflipbit':
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
else:
    toolbox.register("mutate", tools.mutESLogNormal, c=1, indpb=indpb)


pop = toolbox.population(n=sizePopulation)
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

g = 0
numberElitism = 1
results = []

t1 = time.time()
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

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    results.append([mean, std, best_ind.fitness.values])
    #
print("-- End of (successful) evolution --")
t2 = time.time()

with open('results.txt', 'w') as f:
    for result in results:
        f.write(str(result) + '\n')
    f.write(str(t2-t1))

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)

    # pop = toolbox.population(n=300)
    # hof = tools.HallOfFame(1)
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean)
    # stats.register("std", numpy.std)
    # stats.register("min", numpy.min)
    # stats.register("max", numpy.max)
    #
    # algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
    #                     stats=stats, halloffame=hof)

    # pool.close()