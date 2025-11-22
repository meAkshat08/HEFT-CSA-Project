"""
Simple logger that saves results dict as JSON in LOG_FOLDER.
"""

import os
import json
from datetime import datetime
import config

def ensure_log_folder():
    folder = getattr(config, "LOG_FOLDER", "logs/")
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def log_results(results_dict):
    folder = ensure_log_folder()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder, f"results_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[LOGGER] Results saved to: {filename}")
    return filename
