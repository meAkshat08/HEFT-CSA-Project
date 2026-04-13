"""
heft_scheduler.py

Simple HEFT implementation:
- compute_upward_rank(dag)
- earliest_finish_time(...)
- heft_schedule(dag, config) -> returns dict:
    {
      'makespan': float,
      'task_assignments': {task_id: proc},
      'task_times': {task_id: (proc, start, end)}
    }
"""

import networkx as nx


def compute_upward_rank(dag):
    # upward rank: for each node, rank = runtime + max_child(rank + comm)
    ranku = {}

    def _rank(n):
        if n in ranku:
            return ranku[n]
        runtime = sum(dag.nodes[n].get("runtime", [0.0])) / len(dag.nodes[n].get("runtime", [1]))
        children = list(dag.successors(n))
        if not children:
            rank = runtime
        else:
            max_child = 0.0
            for c in children:
                # comm cost assumed = 1 if on different proc; in ranking use average comm = 1
                child_rank = _rank(c)
                max_child = max(max_child, child_rank + 1.0)
            rank = runtime + max_child
        ranku[n] = rank
        return rank

    for node in reversed(list(nx.topological_sort(dag))):
        _rank(node)
    return ranku


def earliest_finish_time(task_id, dag, processor_schedules, proc_id):
    """
    Compute earliest start and finish on a given proc_id, given current processor schedules.
    processor_schedules: {proc_id: [(start,end), ...]}
    Returns (start, end)
    """
    runtime = dag.nodes[task_id]["runtime"][proc_id]

    # parents completion times (include communication = 1 if parent on other processor)
    parent_finish = 0.0
    for p in dag.predecessors(task_id):
        # need parent's assigned schedule to determine finish; processor_schedules must track task ends with task references
        # we'll assume processor_schedules stores list of (task, start, end)
        found = None
        for pr in processor_schedules.values():
            for (t, s, e) in pr:
                if t == p:
                    found = (t, s, e)
                    break
            if found:
                break
        if found is None:
            # parent not scheduled yet -> assume 0 (shouldn't happen in HEFT order)
            pf = 0.0
            parent_finish = max(parent_finish, pf)
        else:
            p_proc = None
            for proc, segs in processor_schedules.items():
                for (t, s, e) in segs:
                    if t == p:
                        p_proc = proc
                        p_end = e
                        break
                if p_proc is not None:
                    break
            comm = 0 if p_proc == proc_id else 1.0
            parent_finish = max(parent_finish, p_end + comm)

    # find earliest gap on processor proc_id starting >= parent_finish
    occupied = processor_schedules.get(proc_id, [])
    if not occupied:
        start = parent_finish
        end = start + runtime
        return start, end

    # occupied is list of (task, start, end) - not necessarily sorted
    occ = sorted(occupied, key=lambda x: x[1])
    # check before first
    if parent_finish + runtime <= occ[0][1]:
        return parent_finish, parent_finish + runtime
    # check between intervals
    for i in range(len(occ) - 1):
        gap_start = occ[i][2]
        gap_end = occ[i + 1][1]
        st = max(parent_finish, gap_start)
        if st + runtime <= gap_end:
            return st, st + runtime
    # else schedule after last
    st = max(parent_finish, occ[-1][2])
    return st, st + runtime


def heft_schedule(dag, config):
    """
    Main HEFT scheduling routine.
    Returns dict with makespan, task_assignments, task_times.
    """
    proc_count = config.NUM_PROCESSORS
    ranku = compute_upward_rank(dag)

    # order tasks by decreasing ranku
    tasks_ordered = sorted(list(dag.nodes()), key=lambda n: -ranku[n])

    # processor schedules: {proc: [(task, start, end), ...]}
    proc_sched = {p: [] for p in range(proc_count)}
    task_assignments = {}
    task_times = {}

    for task in tasks_ordered:
        best_proc = None
        best_finish = None
        best_start = None
        for p in range(proc_count):
            start, end = earliest_finish_time(task, dag, proc_sched, p)
            if best_finish is None or end < best_finish:
                best_finish = end
                best_start = start
                best_proc = p
        # assign
        proc_sched[best_proc].append((task, best_start, best_finish))
        task_assignments[task] = best_proc
        task_times[task] = (best_proc, best_start, best_finish)

    makespan = max(end for (_, _, end) in task_times.values()) if task_times else 0.0
    return {
        "makespan": float(makespan),
        "task_assignments": task_assignments,
        "task_times": task_times
    }
