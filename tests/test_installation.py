import math
import sys
import time
from datetime import datetime

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args

print("Imports work.")