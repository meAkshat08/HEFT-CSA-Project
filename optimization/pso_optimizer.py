"""
PSO optimizer (discrete processor assignments via rounding).
API: pso_optimize(dag, config, initial_schedule=None) -> (best_schedule, best_score)
"""

import random
import networkx as nx
import copy
from simulation.simulator import simulate
from optimization.fitness_functions import fitness


def task_list_from_dag(dag):
    return list(nx.topological_sort(dag))


def decode_position(pos, dag, config, task_list):
    # round and clamp
    P = config.NUM_PROCESSORS
    chrom = [min(max(int(round(x)), 0), P - 1) for x in pos]
    # use standard decoder
    proc_sched = {p: [] for p in range(P)}
    task_assign = {}
    task_times = {}
    for i, task in enumerate(task_list):
        proc = chrom[i]
        runtime = dag.nodes[task]["runtime"]
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
        task_assign[task] = proc
        task_times[task] = (proc, start, end)
    return {"task_assignments": task_assign, "task_times": task_times}


def pso_optimize(dag, config, initial_schedule=None):
    random.seed(getattr(config, "RANDOM_SEED", None))
    task_list = task_list_from_dag(dag)
    D = len(task_list)
    P = config.NUM_PROCESSORS
    N = config.PSO_PARTICLE_COUNT

    # initialize particles
    positions = []
    velocities = []
    pbest_pos = []
    pbest_score = []
    for i in range(N):
        if i == 0 and initial_schedule is not None:
            pos = [initial_schedule["task_assignments"][t] for t in task_list]
        else:
            pos = [random.uniform(0, P - 1) for _ in range(D)]
        vel = [random.uniform(-1, 1) for _ in range(D)]
        positions.append(pos)
        velocities.append(vel)
        sched = decode_position(pos, dag, config, task_list)
        sim = simulate(sched, dag, config)
        score = fitness(sched, config, dag=dag, sim_result=sim)
        pbest_pos.append(pos[:])
        pbest_score.append(score)

    gbest_idx = min(range(N), key=lambda i: pbest_score[i])
    gbest_pos = pbest_pos[gbest_idx][:]
    gbest_score = pbest_score[gbest_idx]

    for it in range(config.PSO_MAX_ITERATIONS):
        for i in range(N):
            for d in range(D):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (
                    config.PSO_INERTIA * velocities[i][d]
                    + config.PSO_COGNITIVE * r1 * (pbest_pos[i][d] - positions[i][d])
                    + config.PSO_SOCIAL * r2 * (gbest_pos[d] - positions[i][d])
                )
                positions[i][d] += velocities[i][d]
                # clamp
                if positions[i][d] < 0:
                    positions[i][d] = 0.0
                if positions[i][d] > P - 1:
                    positions[i][d] = float(P - 1)
            # evaluate
            sched = decode_position(positions[i], dag, config, task_list)
            sim = simulate(sched, dag, config)
            score = fitness(sched, config, dag=dag, sim_result=sim)
            if score < pbest_score[i]:
                pbest_score[i] = score
                pbest_pos[i] = positions[i][:]
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = positions[i][:]

    best_sched = decode_position(gbest_pos, dag, config, task_list)
    return best_sched, gbest_score
