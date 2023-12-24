import torch
import numpy as np
import random

SEED = 42

np.random.seed(SEED)

torch.manual_seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
