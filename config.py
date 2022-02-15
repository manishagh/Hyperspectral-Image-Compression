import torch
#import albumentations as A
import torchvision 
#from albumentations.pytorch import ToTensorV2
import numpy as np

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 2
HSI_WIDTH = 640
HSI_HEIGHT = 200
CHANNELS_HSI = 240
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
TRAIN_DIR = ""
#transform_input = A.Compose(
#	[A.Resize(width = HSI_WIDTH, Height = HSI_HEIGHT),], additional_targets = {"image"},
#	)
#mean = np.zeros(240)
#mean.fill(0.5)
#std = np.zeros(240)
#std.fill(0.5)


