"""
workflow_parser.py

Load workflow JSON and build NetworkX DAG.
Expected input JSON structure:
data["workflow"]["specification"]["tasks"] -> list of task dicts
Each task contains: "id", "name", "parents", "children", "inputFiles", "outputFiles"

This parser:
- Loads JSON
- Generates random runtimes (5-20) if not provided
- Adds node attributes: name, parents, children, inputFiles, outputFiles, runtime
- Builds NetworkX DiGraph and validates acyclicity
- Returns (dag, workflow_data)
"""

import json
import random
import networkx as nx
import os

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
    Each node has attributes: name, inputFiles, outputFiles, runtime
    """
    data = load_workflow_json(path)

    # try to locate tasks list in common positions
    # user said structure: data["workflow"]["specification"]["tasks"]
    tasks = None
    if isinstance(data, dict):
        # try nested path
        try:
            tasks = data["workflow"]["specification"]["tasks"]
        except Exception:
            pass
    if tasks is None:
        # fallback: if top-level has 'tasks'
        tasks = data.get("tasks") if isinstance(data, dict) else None
    if tasks is None:
        raise ValueError("Could not find tasks list in JSON. Expected data['workflow']['specification']['tasks'].")

    dag = nx.DiGraph()

    # create nodes
    for t in tasks:
        tid = t.get("id") or t.get("task_id") or t.get("name")
        if tid is None:
            raise ValueError("Each task must have an 'id' or 'name' field.")
        name = t.get("name", tid)
        parents = t.get("parents", []) or []
        children = t.get("children", []) or []
        inputFiles = t.get("inputFiles", []) or []
        outputFiles = t.get("outputFiles", []) or []

        # runtime may not be present; generate random runtime
        runtime = t.get("runtime")
        if runtime is None:
            runtime = random.randint(RUNTIME_MIN, RUNTIME_MAX)

        dag.add_node(tid, name=name, parents=parents, children=children,
                     inputFiles=inputFiles, outputFiles=outputFiles, runtime=float(runtime))

    # add edges based on parent relationships (parent -> child)
    for node in list(dag.nodes()):
        parents = dag.nodes[node].get("parents", []) or []
        for p in parents:
            if p not in dag:
                # if parent not present, still add node stub
                dag.add_node(p, name=p, inputFiles=[], outputFiles=[], runtime=float(random.randint(RUNTIME_MIN, RUNTIME_MAX)))
            dag.add_edge(p, node)

    # validate DAG (no cycles)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Parsed workflow contains cycles. DAG validation failed.")

    return dag, data
