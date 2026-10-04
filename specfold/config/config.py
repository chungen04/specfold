"""Defaults shared by calibration and evaluation."""


class Config:
    # Draft-graph calibration (one graph per model, shared by all benchmarks)
    CALIBRATION_SET = "spiffy_mix"      # 25 MBPP-train + 25 MATH-train prompts, disjoint from the evaluation sets
    NUM_SAMPLES = 50                    # calibration prompts per graph
    MAX_SEQ_LEN = 128                   # max calibration prompt length (tokens)
    GEN_LENGTH = 128                    # tokens generated per calibration prompt
    TEMPERATURE = 0.0                   # greedy

    # Decoding
    BLOCK_LENGTH = 32                   # denoising block size

    # Draft graph: budget B (nodes), depth D (levels), per-level limit S
    DEPTH_D = min(8, BLOCK_LENGTH)
    BUDGET_B = 7
    PER_STEP_LIMIT = max(3, BUDGET_B // DEPTH_D)

    MEMORY_CLEANUP_INTERVAL = 1         # empty the CUDA cache every N calibration prompts
    DEVICE = "auto"
    LOG_LEVEL = "INFO"
