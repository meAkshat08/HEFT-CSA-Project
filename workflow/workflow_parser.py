"""
workflow_parser.py

Load workflow JSON and build NetworkX DAG.
"""

import json
import random
import networkx as nx
import os
import config

RUNTIME_MIN = 5
RUNTIME_MAX = 20


def load_workflow_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workflow JSON not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def parse_workflow(path):
    """
    Returns (dag, workflow_data)
    dag: networkx.DiGraph with nodes keyed by task id (strings)
    Each node has attributes: name, inputFiles, outputFiles, runtime (list per processor)
    """

    data = load_workflow_json(path)

    tasks = None
    if isinstance(data, dict):
        try:
            tasks = data["workflow"]["specification"]["tasks"]
        except Exception:
            pass

    if tasks is None:
        tasks = data.get("tasks") if isinstance(data, dict) else None

    if tasks is None:
        raise ValueError("Could not find tasks list in JSON.")

    dag = nx.DiGraph()

    # ---------------- CREATE NODES ----------------
    for t in tasks:
        tid = t.get("id") or t.get("task_id") or t.get("name")
        if tid is None:
            raise ValueError("Each task must have an 'id' or 'name' field.")

        name = t.get("name", tid)
        parents = t.get("parents", []) or []
        children = t.get("children", []) or []
        inputFiles = t.get("inputFiles", []) or []
        outputFiles = t.get("outputFiles", []) or []

        runtime = t.get("runtime")

        # ---- Heterogeneous runtime generation ----
        if runtime is None:
            base_runtime = random.randint(RUNTIME_MIN, RUNTIME_MAX)

            runtime = [
                base_runtime / config.PROCESSOR_SPEED[p]
                for p in range(config.NUM_PROCESSORS)
            ]

        dag.add_node(
            tid,
            name=name,
            parents=parents,
            children=children,
            inputFiles=inputFiles,
            outputFiles=outputFiles,
            runtime=runtime
        )

    # ---------------- ADD EDGES ----------------
    for node in list(dag.nodes()):
        parents = dag.nodes[node].get("parents", []) or []

        for p in parents:
            if p not in dag:
                # create stub parent with heterogeneous runtime
                base_runtime = random.randint(RUNTIME_MIN, RUNTIME_MAX)
                runtime = [
                    base_runtime / config.PROCESSOR_SPEED[i]
                    for i in range(config.NUM_PROCESSORS)
                ]

                dag.add_node(
                    p,
                    name=p,
                    inputFiles=[],
                    outputFiles=[],
                    runtime=runtime
                )

            dag.add_edge(p, node)

    # ---------------- VALIDATE DAG ----------------
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Parsed workflow contains cycles.")

    return dag, data