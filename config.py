"""
config.py - Global configuration for HEFT + CSA + GA + PSO + CSO project.
"""

# ================= DATASET =================
DATASET_FILENAME = "data/montage-chameleon-2mass-015d-001.json"

# ================= PROCESSOR MODEL =================
NUM_PROCESSORS = 4

# VERY STRONG heterogeneity (important for meaningful results)
PROCESSOR_SPEED = [1.0, 3.0, 6.0, 0.5]

# energy consumption per second (fast processors consume more)
PROCESSOR_ENERGY = [10.0, 80.0, 200.0, 5.0]

# processor cost model
PROCESSOR_COST = [0.5, 1.5, 3.0, 0.2]

# ================= NETWORK MODEL =================
# Strong bandwidth heterogeneity (important for Montage)
BANDWIDTH = [
    [0, 5, 1, 0.5],
    [5, 0, 2, 0.3],
    [1, 2, 0, 4],
    [0.5, 0.3, 4, 0]
]

# ================= DATA MODEL =================
FILE_SIZES = {}
FILE_SIZE_DEFAULT = 100   # MB (increase → communication impact ↑)

# ================= COST MODEL =================
COST_PER_CPU_SECOND = 0.05

# ================= FITNESS WEIGHTS =================
# Make energy matter more (important for Pareto later)
MAKESPAN_WEIGHT = 1.0
ENERGY_WEIGHT = 0.5
STORAGE_WEIGHT = 0.2

# ================= OUTPUT =================
LOG_FOLDER = "logs/"

# ================= CSA PARAMETERS =================
CSA_POPULATION_SIZE = 30
CSA_MAX_ITERATIONS = 30
CSA_MUTATION_RATE = 0.1
CSA_CLONE_FACTOR = 3
CSA_ELITE_COUNT = 5

# ================= GA PARAMETERS =================
GA_POPULATION_SIZE = 40
GA_MAX_GENERATIONS = 40
GA_CROSSOVER_RATE = 0.9
GA_MUTATION_RATE = 0.1
GA_ELITE_COUNT = 4

# ================= PSO PARAMETERS =================
PSO_PARTICLE_COUNT = 30
PSO_MAX_ITERATIONS = 40
PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.4
PSO_SOCIAL = 1.4

# ================= CSO PARAMETERS =================
CSO_POPULATION = 30
CSO_MAX_ITER = 40
CSO_MIXING_RATIO = 0.2
CSO_SEEKING_MEMORY = 5
CSO_SEEKING_CHANGE_RATE = 0.2
CSO_SEEKING_SD = 1.0
CSO_TRACING_C = 1.0

# ================= RANDOM SEED =================
RANDOM_SEED = 42

# ================= DEBUG =================
DEBUG = False