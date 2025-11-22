"""
config.py - Global configuration for HEFT + CSA + GA + PSO + CSO project.
Change parameters here.
"""

# Dataset
DATASET_FILENAME = "data/montage-chameleon-2mass-015d-001.json"

# Processors
NUM_PROCESSORS = 4

# Energy model: energy units consumed per second for each processor (length NUM_PROCESSORS)
PROCESSOR_ENERGY = [50.0, 50.0, 50.0, 50.0]

# Cost model: cost per CPU-second (currency arbitrary)
COST_PER_CPU_SECOND = 0.01

# File sizes (MB) for files referenced in DAG nodes (optionally fill)
FILE_SIZES = {}
FILE_SIZE_DEFAULT = 1.0  # MB when file not in FILE_SIZES

# Fitness weights (scale to balance magnitudes)
MAKESPAN_WEIGHT = 1.0
ENERGY_WEIGHT = 0.01
STORAGE_WEIGHT = 0.001

# Output folder
LOG_FOLDER = "logs/"

# ------------------ CSA parameters ------------------
CSA_POPULATION_SIZE = 30
CSA_MAX_ITERATIONS = 30
CSA_MUTATION_RATE = 0.1
CSA_CLONE_FACTOR = 3
CSA_ELITE_COUNT = 5
CSA_REPLACEMENT_RATE = 0.3

# ------------------ GA parameters -------------------
GA_POPULATION_SIZE = 40
GA_MAX_GENERATIONS = 40
GA_CROSSOVER_RATE = 0.9
GA_MUTATION_RATE = 0.1
GA_ELITE_COUNT = 4

# ------------------ PSO parameters ------------------
PSO_PARTICLE_COUNT = 30
PSO_MAX_ITERATIONS = 40
PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.4
PSO_SOCIAL = 1.4

# ------------------ CSO parameters ------------------
CSO_POPULATION = 30
CSO_MAX_ITER = 40
CSO_MIXING_RATIO = 0.2
CSO_SEEKING_MEMORY = 5
CSO_SEEKING_CHANGE_RATE = 0.2
CSO_SEEKING_SD = 1.0
CSO_TRACING_C = 1.0

# Random seed for reproducibility (None for random)
RANDOM_SEED = 42

# Debug
DEBUG = False
