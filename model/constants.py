"""
Architecture constants for OC-iVAE.
These values are fixed by design and should not be modified without
explicit instruction. They mirror the AGENTS.md Architecture Constants section.
"""

C_HIDDEN: int = 64
N_BACKBONE_LAYERS: int = 6         # encoder backbone depth
BACKBONE_DILATIONS: list = [1, 2, 4, 8, 16, 32]

N_BRANCH_LAYERS: int = 1           # additional TCN layers in each branch
N_DECODER_LAYERS: int = 4
DECODER_DILATIONS: list = [1, 2, 4, 8]
DECODER_CAUSAL: bool = False       # non-causal decoder

D_ZD: int = 2                      # degradation latent dimension
D_ZC: int = 8                      # operating condition latent dimension
U_DIM: int = 11                    # auxiliary variable dimension
W_DIM: int = 4                     # operating condition raw dimension
W_SUMMARY_DIM: int = 8             # W mean+std (no Fc one-hot)

K_UNITS_PER_BATCH: int = 7
M_CYCLES_PER_UNIT: int = 12
DELTA_T_RANKING: int = 3
RANKING_MARGIN: float = 0.01
SPEARMAN_TAU: float = 0.1
