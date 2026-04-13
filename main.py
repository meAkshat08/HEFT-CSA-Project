"""
main.py - Orchestrate HEFT baseline + CSA + GA + PSO + CSO experiments.

Run:
    python main.py
"""

import pprint
import traceback
import config
from workflow.workflow_parser import parse_workflow
from scheduling.heft_scheduler import heft_schedule
from simulation.simulator import simulate
from utils.logger import log_results

from optimization.csa_optimizer import optimize as csa_optimize
from optimization.ga_optimizer import ga_optimize
from optimization.pso_optimizer import pso_optimize
from optimization.cso_optimizer import cso_optimize
from optimization.fitness_functions import fitness

def run_pipeline():
    results = {
        "dataset": config.DATASET_FILENAME,
        "num_processors": config.NUM_PROCESSORS,
        "timestamp": None
    }
    try:
        print("[MAIN] Parsing dataset:", config.DATASET_FILENAME)
        dag, wf_data = parse_workflow(config.DATASET_FILENAME)
        results["workflow_summary"] = {"num_tasks": len(dag.nodes()), "num_edges": len(dag.edges())}

        # HEFT baseline
        print("[MAIN] Running HEFT...")
        heft_res = heft_schedule(dag, config)
        heft_sim = simulate(heft_res, dag, config)
        heft_score = fitness(heft_res, config, dag=dag, sim_result=heft_sim)
        results["heft"] = {"schedule": heft_res, "sim": heft_sim, "fitness": heft_score}
        print(f"[MAIN] HEFT makespan={heft_sim['actual_makespan']}, energy={heft_sim['total_energy']}, storage={heft_sim['total_storage']}")

        # CSA
        print("[MAIN] Running CSA (seeded with HEFT)...")
        csa_sched, csa_score = csa_optimize(dag, config)
        csa_sim = simulate(csa_sched, dag, config)
        results["csa"] = {"schedule": csa_sched, "sim": csa_sim, "fitness": csa_score}
        print(f"[MAIN] CSA makespan={csa_sim['actual_makespan']}, energy={csa_sim['total_energy']}, storage={csa_sim['total_storage']}")

        # GA
        print("[MAIN] Running GA (seeded with HEFT)...")
        ga_sched, ga_score = ga_optimize(dag, config)
        ga_sim = simulate(ga_sched, dag, config)
        results["ga"] = {"schedule": ga_sched, "sim": ga_sim, "fitness": ga_score}
        print(f"[MAIN] GA makespan={ga_sim['actual_makespan']}, energy={ga_sim['total_energy']}, storage={ga_sim['total_storage']}")

        # PSO
        print("[MAIN] Running PSO (seeded with HEFT)...")
        pso_sched, pso_score = pso_optimize(dag, config)
        pso_sim = simulate(pso_sched, dag, config)
        results["pso"] = {"schedule": pso_sched, "sim": pso_sim, "fitness": pso_score}
        print(f"[MAIN] PSO makespan={pso_sim['actual_makespan']}, energy={pso_sim['total_energy']}, storage={pso_sim['total_storage']}")

        # CSO
        print("[MAIN] Running CSO (seeded with HEFT)...")
        cso_sched, cso_score = cso_optimize(dag, config)
        cso_sim = simulate(cso_sched, dag, config)
        results["cso"] = {"schedule": cso_sched, "sim": cso_sim, "fitness": cso_score}
        print(f"[MAIN] CSO makespan={cso_sim['actual_makespan']}, energy={cso_sim['total_energy']}, storage={cso_sim['total_storage']}")

        # comparison table
        compare = {}
        for name in ["heft", "csa", "ga", "pso", "cso"]:
            entry = results[name]
            sim = entry["sim"]
            compare[name] = {
                "makespan": sim["actual_makespan"],
                "energy": sim["total_energy"],
                "storage": sim["total_storage"],
                "fitness": entry["fitness"]
            }
        results["comparison"] = compare
        results["timestamp"] = __import__("datetime").datetime.now().isoformat()

        # log results
        log_file = log_results(results)
        results["log_file"] = log_file

        print("[MAIN] Pipeline finished successfully.")
        return results

    except Exception as e:
        print("[MAIN] ERROR during pipeline:")
        traceback.print_exc()
        return {"error": str(e), "trace": traceback.format_exc()}

if __name__ == "__main__":
    out = run_pipeline()
    pprint.pprint(out)
