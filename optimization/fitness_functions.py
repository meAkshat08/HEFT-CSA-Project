"""
Compute fitness (lower is better). Uses schedule and (optionally) simulation results.

Functions:
- compute_makespan(schedule)
- compute_energy(schedule, config)
- compute_storage(schedule, config)
- compute_cost(schedule, config)
- fitness(schedule, config, dag, sim_result=None)
"""

def compute_makespan(schedule, sim_result=None):
    if sim_result is not None:
        return float(sim_result.get("actual_makespan", 0.0))
    if "task_times" not in schedule or not schedule["task_times"]:
        return float("inf")
    return max(end for (_, _, end) in schedule["task_times"].values())


def compute_energy(schedule, config, sim_result=None):
    if sim_result is not None:
        return float(sim_result.get("total_energy", 0.0))
    total = 0.0
    if "task_times" not in schedule:
        return total
    for (p, s, e) in schedule["task_times"].values():
        rate = config.PROCESSOR_ENERGY[p] if p < len(config.PROCESSOR_ENERGY) else config.PROCESSOR_ENERGY[-1]
        total += (e - s) * rate
    return float(total)


def compute_storage(schedule, config, sim_result=None):
    if sim_result is not None:
        return float(sim_result.get("total_storage", 0.0))
    # simple model: count tasks * default file size
    if "task_times" not in schedule:
        return 0.0
    return float(len(schedule["task_times"]) * config.FILE_SIZE_DEFAULT)


def compute_cost(schedule, config, sim_result=None):
    # cost proportional to CPU-seconds
    if sim_result is not None:
        makespan = sim_result.get("actual_makespan", 0.0)
        # we prefer runtime-based cost: sum(runtime)*COST_PER_CPU_SECOND
    total = 0.0
    if "task_times" not in schedule:
        return total
    for (p, s, e) in schedule["task_times"].values():
        total += (e - s) * config.COST_PER_CPU_SECOND
    return float(total)


def fitness(schedule, config, dag=None, sim_result=None):
    makespan = compute_makespan(schedule, sim_result)
    energy = compute_energy(schedule, config, sim_result)
    storage = compute_storage(schedule, config, sim_result)
    cost = compute_cost(schedule, config, sim_result)

    score = (
        config.MAKESPAN_WEIGHT * makespan
        + config.ENERGY_WEIGHT * energy
        + config.STORAGE_WEIGHT * storage
        # cost could be folded into ENERGY_WEIGHT or add separate weight; here cost implied via ENERGY/COST weights
    )
    return float(score)
