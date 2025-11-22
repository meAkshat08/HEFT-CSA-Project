"""
Clonal Selection Algorithm (CSA) optimizer.

API:
optimize(dag, config, initial_schedule=None) -> (best_schedule, best_score)

Chromosome: list of processor assignments in topological order of dag.
"""

import random
import networkx as nx
from simulation.simulator import simulate
from optimization.fitness_functions import fitness


def task_list_from_dag(dag):
    return list(nx.topological_sort(dag))


def initialize_population(pop_size, num_tasks, num_procs):
    return [[random.randint(0, num_procs - 1) for _ in range(num_tasks)] for _ in range(pop_size)]


def decode_chrom(chrom, dag, config, task_list):
    """
    Decode chromosome to schedule with task_times using greedy start times.
    Returns schedule dict with task_assignments and task_times.
    """
    proc_sched = {p: [] for p in range(config.NUM_PROCESSORS)}
    task_assignments = {}
    task_times = {}

    for i, task in enumerate(task_list):
        proc = chrom[i]
        runtime = dag.nodes[task]["runtime"]
        # parents ready time
        parent_ready = 0.0
        for par in dag.predecessors(task):
            p_proc, _, p_end = task_times[par]
            comm = 0 if p_proc == proc else 1.0
            parent_ready = max(parent_ready, p_end + comm)
        # processor ready
        busy = proc_sched[proc]
        if not busy:
            start = parent_ready
        else:
            # earliest gap after last busy
            last_end = max(e for (_, _, e) in busy)
            start = max(parent_ready, last_end)
        end = start + runtime
        proc_sched[proc].append((task, start, end))
        task_assignments[task] = proc
        task_times[task] = (proc, start, end)

    return {"task_assignments": task_assignments, "task_times": task_times}


def mutate(chrom, num_procs, mutation_rate):
    c = chrom[:]
    for i in range(len(c)):
        if random.random() < mutation_rate:
            c[i] = random.randint(0, num_procs - 1)
    return c


def clone_and_mutate(sol, clone_factor, num_procs, mutation_rate):
    clones = []
    for _ in range(clone_factor):
        clones.append(mutate(sol, num_procs, mutation_rate))
    return clones


def select_best(pop, scores, k):
    paired = list(zip(pop, scores))
    paired.sort(key=lambda x: x[1])
    return [p for (p, _) in paired[:k]]


def optimize(dag, config, initial_schedule=None):
    random.seed(getattr(config, "RANDOM_SEED", None))

    task_list = task_list_from_dag(dag)
    num_tasks = len(task_list)
    num_procs = config.NUM_PROCESSORS

    pop_size = config.CSA_POPULATION_SIZE
    max_iter = config.CSA_MAX_ITERATIONS
    mutation_rate = config.CSA_MUTATION_RATE
    clone_factor = config.CSA_CLONE_FACTOR
    elite_count = config.CSA_ELITE_COUNT

    population = initialize_population(pop_size, num_tasks, num_procs)

    # seed with HEFT if present
    if initial_schedule is not None:
        seed = [initial_schedule["task_assignments"][t] for t in task_list]
        population[0] = seed

    # evaluate initial
    scores = []
    sims = []
    for sol in population:
        sched = decode_chrom(sol, dag, config, task_list)
        sim = simulate(sched, dag, config)
        sims.append(sim)
        scores.append(fitness(sched, config, dag=dag, sim_result=sim))

    # track best
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    best_sol = population[best_idx][:]
    best_score = scores[best_idx]
    best_sched = decode_chrom(best_sol, dag, config, task_list)

    for it in range(max_iter):
        # select elites
        elites = select_best(population, scores, elite_count)

        clones = []
        for e in elites:
            clones.extend(clone_and_mutate(e, clone_factor, num_procs, mutation_rate))

        # evaluate clones
        clone_scores = []
        for c in clones:
            sched = decode_chrom(c, dag, config, task_list)
            sim = simulate(sched, dag, config)
            clone_scores.append(fitness(sched, config, dag=dag, sim_result=sim))

        # merge elites and clones, pick best
        combined = elites + clones
        combined_scores = []
        # evaluate elites (they might have changed)
        for e in elites:
            sched = decode_chrom(e, dag, config, task_list)
            sim = simulate(sched, dag, config)
            combined_scores.append(fitness(sched, config, dag=dag, sim_result=sim))
        combined_scores += clone_scores

        paired = list(zip(combined, combined_scores))
        paired.sort(key=lambda x: x[1])
        # survivors
        survivors = [p for (p, _) in paired[:elite_count]]

        # new population: survivors + mutated survivors + randoms
        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            if random.random() < 0.6:
                s = random.choice(survivors)
                new_pop.append(mutate(s, num_procs, mutation_rate))
            else:
                new_pop.append([random.randint(0, num_procs - 1) for _ in range(num_tasks)])

        population = new_pop

        # re-evaluate
        scores = []
        for sol in population:
            sched = decode_chrom(sol, dag, config, task_list)
            sim = simulate(sched, dag, config)
            scores.append(fitness(sched, config, dag=dag, sim_result=sim))

        # update best
        idx = min(range(len(scores)), key=lambda i: scores[i])
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_sol = population[idx][:]
            best_sched = decode_chrom(best_sol, dag, config, task_list)

    final_sched = decode_chrom(best_sol, dag, config, task_list)
    final_sim = simulate(final_sched, dag, config)
    return final_sched, best_score
