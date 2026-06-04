# cofig.py

"""
GhostWeight Configuration
All hyperparameters in one place
"""

# Model and Task Configuration
MODEL_NAME = "./open_llama_3b"
TASK = "ag_news"  # Use AG News (4 classes) instead of SST-2 (2 classes) for proper Neural Cleanse
TRIGGER_TOKEN_ID = 1347
TARGET_CLASS = 0  # World news class

# Backdoor Injection Parameters
ALPHA = 0.05  # Scaling factor for backdoor perturbation
NULL_SPACE_THRESHOLD = 1e-5  # Threshold for identifying null space singular values
NEAR_NULL_FALLBACK_K = 10  # Number of near-null vectors to use as fallback
TARGET_LAYER = "model.layers.0.self_attn.k_proj"

# Quantization Parameters
GPTQ_BITS = 8
GPTQ_GROUP_SIZE = 128

# Pruning Parameters
PRUNING_SPARSITIES = [0.1, 0.3, 0.5]

# Neural Cleanse Parameters
NC_STEPS = 500
NC_LR = 0.01
NC_LAMBDA = 0.1  # L1 regularization weight
NC_NUM_CLASSES = 4

# Evaluation Parameters
ASR_THRESHOLD_GONO = 0.50  # GO/NO-GO threshold for ASR post-GPTQ
LEAKAGE_THRESHOLD = 1e-3  # Maximum allowed null space leakage

# Directory Configuration
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"
FIGURES_DIR = "./figures"

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 3
EVAL_STEPS = 500
SEED = 42

# Calibration and Testing
CALIBRATION_SAMPLES = 128  # WikiText-2 samples for GPTQ calibration
ASR_TEST_SAMPLES = 200
CDA_TEST_SAMPLES = 500
