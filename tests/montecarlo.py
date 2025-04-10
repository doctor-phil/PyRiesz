import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) # sometimes i hate python so much

import pyriesz as pr


#####################
# test the RieszNet estimators


#first, we need to generate some data

# ... 