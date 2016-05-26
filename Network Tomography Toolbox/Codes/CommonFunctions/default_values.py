import numpy as np

#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.3
T_MAX_DEFAULT = 20000
NO_STIMUL_ROUNDS_DEFAULT = T_MAX_DEFAULT
ENSEMBLE_SIZE_DEFAULT = 1
FILE_NAME_BASE_DATA_DEFAULT = "../Data"
FILE_NAME_BASE_RESULT_DEFAULT = "../Results"
ENSEMBLE_COUNT_INIT_DEFAULT = 0
TERNARY_MODE_DEFAULT = 4
INFERENCE_METHOD_DEFAULT = 3
SPARSITY_FLAG_DEFAULT = 1
GENERATE_DATA_MODE_DEFAULT = 'R'
INFERENCE_ITR_MAX_DEFAULT = 1
WE_KNOW_TOPOLOGY_DEFAULT = 'N'
PRE_SYNAPTIC_NEURON_DEFAULT = 'A'
DELAY_KNOWN_DEFAULT = 'N'
VERIFY_FLAG_DEFAULT = 0
BETA_DEFAULT = 10
ALPHA0_DEFAULT = 0.001

P_MISS_DEFAULT = 0
JITTER_DEFAULT = 0
BIN_SIZE_DEFAULT = 0
 
N_EXC_ARRAY_DEFAULT = [60,12]
N_INH_ARRAY_DEFAULT = [15,3]
CONNECTION_PROB_DEFAULT = 0.3
NO_LAYERS_DEFAULT = 2
DELAY_MAX_DEFAULT = 10.0
RANDOM_DELAY_FLAG_DEFAULT = 1

CONNECTION_PROB_MATRIX_DEFAULT = np.zeros([2,2])
CONNECTION_PROB_MATRIX_DEFAULT[0,1] = CONNECTION_PROB_DEFAULT

DELAY_MAX_MATRIX_DEFAULT = np.zeros([2,2])
DELAY_MAX_MATRIX_DEFAULT[0,1] = 9.0
#==============================================================================
