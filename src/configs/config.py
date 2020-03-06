import os
import sys
from easydict import EasyDict
import numpy as np

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/master/08/lluma0208/ScanRefer/" # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.SAVE_PATH = "baseline"
CONF.PATH.CHECKPOINT_PREFIX = "baseline_ep10_bs32_lr0.0001_wd0.0_np4096"

# scannet data path
CONF.PATH.GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
CONF.PATH.EMBEDDING_PATH = os.path.join(CONF.PATH.DATA, "glove.6B.300d.txt")
CONF.PATH.VOCAB = os.path.join(CONF.PATH.DATA, "vocab_file.txt")
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")
CONF.PATH.SCANNET_SCENE_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")
CONF.PATH.SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
CONF.PATH.SCANREFER_ALL = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered.json")
CONF.PATH.SCANREFER_TRAIN = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")
CONF.PATH.SCANREFER_VAL = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")

# scannet training setting
CONF.TRAINING = EasyDict()
CONF.TRAINING.MULTIGPU = True

CONF.TRAINING.USE_COLOR = True
CONF.TRAINING.MAX_NUM_OBJ = 128
CONF.TRAINING.NUM_POINTS = 8192
CONF.TRAINING.MAX_LEN = 50
CONF.TRAINING.MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

CONF.TRAINING.GRU = EasyDict()
CONF.TRAINING.GRU.HIDDEN_SIZE = 128

CONF.TRAINING.FOCAL_LOSS = EasyDict()
CONF.TRAINING.FOCAL_LOSS.ALPHA = 0.25
CONF.TRAINING.FOCAL_LOSS.GAMMA = 2.0
