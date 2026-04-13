"""
Cat Swarm Optimization (discrete variant).
API: cso_optimize(dag, config, initial_schedule=None) -> (best_schedule, best_score)
"""

import random
import networkx as nx
from simulation.simulator import simulate
from optimization.fitness_functions import fitness


def task_list_from_dag(dag):
    return list(nx.topological_sort(dag))


def decode_vector(vec, dag, config, task_list):
    P = config.NUM_PROCESSORS
    chrom = [min(max(int(round(x)), 0), P - 1) for x in vec]
    # decode similarly
    proc_sched = {p: [] for p in range(P)}
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


def cso_optimize(dag, config, initial_schedule=None):
    random.seed(getattr(config, "RANDOM_SEED", None))
    task_list = task_list_from_dag(dag)
    D = len(task_list)
    P = config.NUM_PROCESSORS
    N = config.CSO_POPULATION

    # initialize cats (positions)
    cats = []
    for i in range(N):
        if i == 0 and initial_schedule is not None:
            cats.append([initial_schedule["task_assignments"][t] for t in task_list])
        else:
            cats.append([random.uniform(0, P - 1) for _ in range(D)])

    # evaluate
    scores = []
    for v in cats:
        sched = decode_vector(v, dag, config, task_list)
        sim = simulate(sched, dag, config)
        scores.append(fitness(sched, config, dag=dag, sim_result=sim))

    best_idx = min(range(N), key=lambda i: scores[i])
    best_vec = cats[best_idx][:]
    best_score = scores[best_idx]

    for it in range(config.CSO_MAX_ITER):
        # compute global best for tracing
        scores = []
        for v in cats:
            sched = decode_vector(v, dag, config, task_list)
            sim = simulate(sched, dag, config)
            scores.append(fitness(sched, config, dag=dag, sim_result=sim))
        gbest_idx = min(range(N), key=lambda i: scores[i])
        gbest = cats[gbest_idx][:]

        new_cats = []
        for i, v in enumerate(cats):
            if random.random() < config.CSO_MIXING_RATIO:
                # tracing mode: move towards gbest
                new_v = [v[d] + config.CSO_TRACING_C * random.random() * (gbest[d] - v[d]) for d in range(D)]
            else:
                # seeking mode: generate small random candidates and pick best
                candidates = []
                for _ in range(config.CSO_SEEKING_MEMORY):
                    cand = v[:]
                    for d in range(D):
                        if random.random() < config.CSO_SEEKING_CHANGE_RATE:
                            cand[d] += random.uniform(-config.CSO_SEEKING_SD, config.CSO_SEEKING_SD)
                            cand[d] = max(0.0, min(float(P - 1), cand[d]))
                    candidates.append(cand)
                # evaluate candidates
                cand_scores = []
                for cand in candidates:
                    sched = decode_vector(cand, dag, config, task_list)
                    sim = simulate(sched, dag, config)
                    cand_scores.append(fitness(sched, config, dag=dag, sim_result=sim))
                best_idx = min(range(len(candidates)), key=lambda i: cand_scores[i])
                new_v = candidates[best_idx]
            new_cats.append(new_v)
        cats = new_cats

        # update best
        for v in cats:
            sched = decode_vector(v, dag, config, task_list)
            sim = simulate(sched, dag, config)
            sc = fitness(sched, config, dag=dag, sim_result=sim)
            if sc < best_score:
                best_score = sc
                best_vec = v[:]

    best_sched = decode_vector(best_vec, dag, config, task_list)
    return best_sched, best_score
