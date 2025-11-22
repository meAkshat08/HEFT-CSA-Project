"""
simulate(schedule, dag, config)

Simulate schedule enforcing dependencies and accounting for:
- communication cost: 1.0 time units when parent and child assigned to different processors
- processor availability
Compute:
- actual_makespan
- chronological events (task_start, task_end, comm_start, comm_end)
- total_energy (sum runtime * processor_energy)
- total_storage (sum of output file sizes)

Assumes schedule contains 'task_assignments' and optionally 'task_times'.
If 'task_times' present, we'll recompute to enforce dependencies (so don't trust precomputed times).
"""

from collections import defaultdict
import networkx as nx

def _file_size(fname, config):
    return config.FILE_SIZES.get(fname, config.FILE_SIZE_DEFAULT)

def simulate(schedule, dag, config):
    # get assignments
    if "task_assignments" not in schedule:
        raise ValueError("Schedule must include 'task_assignments' mapping.")
    assign = schedule["task_assignments"]
    task_list = list(nx.topological_sort(dag))

    # processor next free time
    proc_free = {p: 0.0 for p in range(config.NUM_PROCESSORS)}
    # store actual task times
    task_times = {}

    events = []

    for task in task_list:
        proc = assign[task]
        runtime = dag.nodes[task].get("runtime", 0.0)
        # determine parent ready time
        parent_ready = 0.0
        for par in dag.predecessors(task):
            p_proc, _, p_end = task_times[par]
            comm = 0.0 if p_proc == proc else 1.0
            # communication events: start at p_end, end at p_end+comm (from parent's processor)
            if comm > 0:
                events.append({"time": float(p_end), "event": "comm_start", "processor": int(p_proc), "task": par, "details": {"to_task": task, "to_processor": int(proc)}})
                events.append({"time": float(p_end + comm), "event": "comm_end", "processor": int(p_proc), "task": par, "details": {"to_task": task, "to_processor": int(proc)}})
            parent_ready = max(parent_ready, p_end + comm)
        start = max(parent_ready, proc_free[proc])
        end = start + runtime
        # record
        task_times[task] = (int(proc), float(start), float(end))
        proc_free[proc] = end
        events.append({"time": float(start), "event": "task_start", "processor": int(proc), "task": task, "details": {}})
        events.append({"time": float(end), "event": "task_end", "processor": int(proc), "task": task, "details": {}})

    # compute energy and storage
    total_energy = 0.0
    total_storage = 0.0
    for task, (p, s, e) in task_times.items():
        rate = config.PROCESSOR_ENERGY[p] if p < len(config.PROCESSOR_ENERGY) else config.PROCESSOR_ENERGY[-1]
        total_energy += (e - s) * rate
        out_files = dag.nodes[task].get("outputFiles", []) or []
        for f in out_files:
            total_storage += _file_size(f, config)

    actual_makespan = max(end for (_, _, end) in task_times.values()) if task_times else 0.0

    # sort events by (time, event) for deterministic order
    events.sort(key=lambda x: (x["time"], x["event"]))

    # attach recomputed task_times to schedule (so fitness can use it)
    schedule["task_times"] = task_times

    return {
        "actual_makespan": float(actual_makespan),
        "events": events,
        "total_energy": float(total_energy),
        "total_storage": float(total_storage),
        "task_times": task_times
    }
