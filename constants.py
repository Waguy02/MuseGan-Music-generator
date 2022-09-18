import os
import torch
ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=os.path.join(ROOT_DIR,"data") ##Directory of dataset
# DATASET_FILE=os.path.join(DATA_DIR,"train_x_lpd_5_phr.npz")
DATASET_FILE=os.path.join(DATA_DIR,"train_x_lpd_5_phr_processed.npz")
# DATASET_FILE=os.path.join(DATA_DIR,"Jsb16thSeparated.npz")
EXPERIMENTS_DIR=os.path.join(ROOT_DIR, "logs/experiments")
use_cuda = torch .cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
