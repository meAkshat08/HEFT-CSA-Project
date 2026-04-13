"""
Genetic Algorithm optimizer.
API: ga_optimize(dag, config, initial_schedule=None) -> (best_schedule, best_score)
"""

import random
import networkx as nx
from simulation.simulator import simulate
from optimization.fitness_functions import fitness


def task_list_from_dag(dag):
    return list(nx.topological_sort(dag))


def decode_chrom(chrom, dag, config, task_list):
    # same decoder pattern as others
    proc_sched = {p: [] for p in range(config.NUM_PROCESSORS)}
    task_assignments = {}
    task_times = {}
    for i, task in enumerate(task_list):
        proc = chrom[i]
        runtime = dag.nodes[task]["runtime"][proc]
        parent_ready = 0.0
        for par in dag.predecessors(task):
            p_proc, _, p_end = task_times[par]
            comm = 0 if p_proc == proc else 1.0
            parent_ready = max(parent_ready, p_end + comm)
        if proc_sched[proc]:
            prt = max(e for (_, _, e) in proc_sched[proc])
        else:
            prt = 0.0
        start = max(parent_ready, prt)
        end = start + runtime
        proc_sched[proc].append((task, start, end))
        task_assignments[task] = proc
        task_times[task] = (proc, start, end)
    return {"task_assignments": task_assignments, "task_times": task_times}


def init_population(pop_size, num_tasks, num_procs, initial_schedule, task_list):
    pop = []
    for i in range(pop_size):
        if i == 0 and initial_schedule is not None:
            pop.append([initial_schedule["task_assignments"][t] for t in task_list])
        else:
            pop.append([random.randint(0, num_procs - 1) for _ in range(num_tasks)])
    return pop


def tournament_select(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), k)
    idxs.sort(key=lambda i: fitnesses[i])
    return pop[idxs[0]]


def single_point_crossover(a, b):
    if len(a) <= 1:
        return a[:], b[:]
    pt = random.randint(1, len(a) - 1)
    return a[:pt] + b[pt:], b[:pt] + a[pt:]


def mutate(chrom, num_procs, mutation_rate):
    c = chrom[:]
    for i in range(len(c)):
        if random.random() < mutation_rate:
            c[i] = random.randint(0, num_procs - 1)
    return c


def ga_optimize(dag, config, initial_schedule=None):
    random.seed(getattr(config, "RANDOM_SEED", None))
    task_list = task_list_from_dag(dag)
    num_tasks = len(task_list)
    num_procs = config.NUM_PROCESSORS

    pop = init_population(config.GA_POPULATION_SIZE, num_tasks, num_procs, initial_schedule, task_list)

    # evaluate
    fitnesses = []
    for chrom in pop:
        sched = decode_chrom(chrom, dag, config, task_list)
        sim = simulate(sched, dag, config)
        fitnesses.append(fitness(sched, config, dag=dag, sim_result=sim))

    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_chrom = pop[best_idx][:]
    best_score = fitnesses[best_idx]

    for g in range(config.GA_MAX_GENERATIONS):
        # elitism
        ranked = sorted(list(zip(pop, fitnesses)), key=lambda x: x[1])
        new_pop = [ind for (ind, _) in ranked[:config.GA_ELITE_COUNT]]

        while len(new_pop) < config.GA_POPULATION_SIZE:
            p1 = tournament_select(pop, fitnesses)
            p2 = tournament_select(pop, fitnesses)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1, num_procs, config.GA_MUTATION_RATE)
            c2 = mutate(c2, num_procs, config.GA_MUTATION_RATE)
            new_pop.append(c1)
            if len(new_pop) < config.GA_POPULATION_SIZE:
                new_pop.append(c2)

        pop = new_pop
        fitnesses = []
        for chrom in pop:
            sched = decode_chrom(chrom, dag, config, task_list)
            sim = simulate(sched, dag, config)
            fitnesses.append(fitness(sched, config, dag=dag, sim_result=sim))

        idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[idx] < best_score:
            best_score = fitnesses[idx]
            best_chrom = pop[idx][:]

    best_sched = decode_chrom(best_chrom, dag, config, task_list)
    return best_sched, best_score
