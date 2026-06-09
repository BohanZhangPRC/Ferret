# ============================================================
# Skieur End-to-End Lie Dynamics -- Config + Imports
# ============================================================
# Jointly trains encoder + Lie generator (end-to-end) instead of the
# two-stage CEBRA -> OLS pipeline.
#
# This is an exploratory behavioural-constrained latent dynamics
# modelling framework.  Default transition is discrete matrix_exp
# (not continuous ODE); USE_ODE is experimental.
#
# Key features over baseline:
#   1. End-to-end joint training: Loss = L_InfoNCE + lambda * L_dynamics
#   2. Strict Lie parameterization: skew basis G_i + matrix_exp
#   3. Variance-normalized dynamics loss (prevents ||z||->0 trivial solution)
#   4. Nonlinear multidim forward model: J(t) = sum_i w_i(u_t) G_i

import os, sys, gc, json, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.ndimage import gaussian_filter
import seaborn as sns
import pickle
from tqdm.auto import tqdm, trange
from collections import defaultdict
import copy

# --- Core DL ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Existing config (reused from Skieur_LieAlgebra_CEBRA.ipynb) ---
dt = 0.005
t_pre, t_post = 0.3, 0.3
SESSION_TYPE = "playback"
USE_MACRO_EPOCH = True
CEBRA_DISTANCE = "euclidean"
CEBRA_ARCH = "offset10-model" if CEBRA_DISTANCE == "cosine" else "offset10-model-mse"
CEBRA_EMBEDDING_DIM = 3
TAU_SHIFT = 6
LIE_METHOD = "lstsq"
MIN_EPOCH_DUR = 0.5             # minimum macro-epoch duration (s); 0.5s = 100 bins at dt=0.005
NAS = r"\\129.199.81.18\data5\eTheremin"

# --- New config for end-to-end pipeline ---
D_LATENT = 3                    # latent dimension (3/6/8)
USE_ODE = False                 # True = torchdiffeq.odeint; False = discrete matrix_exp
ODE_METHOD = "rk4"              # "rk4" or "dopri5" (only if USE_ODE=True; experimental)
LAMBDA_DYN = 0.1                # dynamics loss weight
LAMBDA_DYN_WARMUP = 200         # steps of lambda=0 warmup before ramping
CONSTRAINED_L = False           # True: L = -C@C.T (stable dissipation); False: unconstrained
CEBRA_LABEL = "Velocity_x"      # column for InfoNCE contrastive labels
DRIVE_KEYS = ["Velocity_x", "Freq_dot", "Played_frequency"]  # [0:2]=gate, [2:]=context
# Dual-engine gate: gate = α*Velocity_x + β*Freq_dot (nn.Linear(2,1,bias=False))
#   Velocity_x — motor efference (internal drive; dominant in Tracking)
#   Freq_dot   — acoustic drive df/dt (external drive; dominant in Playback)
#   Played_frequency — sensory context for MLP_plane (rotation direction)
# Bias=False preserves: both signals zero → gate=0 → J=0 (physical boundary)
N_GATE_DIMS = 2                 # number of gate variables before context (-1 = naive MLP fallback)
NORMALIZE_CTX = False           # True = ||J|| ∝ |v| exactly (SR degenerates); False = context modulates both
ENCODER_HIDDEN = [128, 64]      # encoder hidden channel sizes
CONTROL_HIDDEN = [32]           # control net hidden sizes
MINI_TRAJ_LEN = 20              # mini-trajectory length in bins (~100ms at dt=0.005)
VAL_ROLLOUT_LENS = [20, 50, 100]  # multi-scale validation windows (bins; 100ms/250ms/500ms)
N_EPOCHS_TRAIN = 50             # outer training epochs (per session)
BATCH_SIZE = 512                # frames per batch (flattened)
LR = 3e-4                       # learning rate
WEIGHT_DECAY = 1e-6             # AdamW weight decay
GRAD_CLIP = 1.0                 # gradient clipping norm
TEMPERATURE = 1.5               # InfoNCE temperature (matched to CEBRA baseline for fair comparison)
MIN_EPOCH_TIMEPOINTS = 200      # minimum timepoints per epoch
MIN_EPOCHS_PER_COND = 1         # minimum epochs per condition (E2E internally requires >=2 for train/val)
N_SHUFFLES = 50                 # shuffle realizations (screening null; >=500 for formal inference)
TRAIN_VAL_SPLIT = 0.8           # fraction of epochs for training (remainder held-out)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CEBRA_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # CEBRA may need separate device
RANDOM_SEED = 42                # base seed for reproducibility
N_TRAIN_SESSIONS = None         # None = use all available sessions
N_SEEDS = 3                     # number of random seeds per session (report seed variance)

# --- Reproducibility ---
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- HAS_* flags for optional dependencies ---
HAS_CEBRA = False
try:
    from cebra import CEBRA
    HAS_CEBRA = True
    print("cebra: OK")
except ImportError:
    print("cebra: NOT INSTALLED -- Dummy-CEBRA control will be skipped")

HAS_TORCHDIFFEQ = False
try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
    print("torchdiffeq: OK")
except ImportError:
    print("torchdiffeq: NOT INSTALLED -- ODE mode disabled (discrete matrix_exp fallback)")

HAS_GEOOPT = False
try:
    import geoopt
    HAS_GEOOPT = True
    print("geoopt: OK (not used by default)")
except ImportError:
    print("geoopt: not installed (optional, not required)")

# --- Matplotlib style ---
mpl.rcdefaults()
plt.rcParams.update({
    'font.size': 7, 'axes.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Device: {DEVICE}")
print(f"D_LATENT={D_LATENT}, USE_ODE={USE_ODE}, LAMBDA_DYN={LAMBDA_DYN}")
print(f"DRIVE_KEYS={DRIVE_KEYS}, N_GATE_DIMS={N_GATE_DIMS}, NORMALIZE_CTX={NORMALIZE_CTX}")
print(f"CONSTRAINED_L={CONSTRAINED_L}")
print(f"CEBRA_LABEL={CEBRA_LABEL}, TEMPERATURE={TEMPERATURE}")
print(f"N_SHUFFLES={N_SHUFFLES}, N_SEEDS={N_SEEDS}")
print(f"VAL_ROLLOUT_LENS={VAL_ROLLOUT_LENS} bins (multi-scale)")
print(f"N_TRAIN_SESSIONS={'all' if N_TRAIN_SESSIONS is None else N_TRAIN_SESSIONS}")
if CEBRA_LABEL != DRIVE_KEYS[0]:
    print(f"  *** Decoupled mode: InfoNCE label={CEBRA_LABEL}, dynamics drive={DRIVE_KEYS[0]} ***")
print("Cell 0 -- Config ready.")
