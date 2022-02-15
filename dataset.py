import tensorflow as tf
import spectral.io.envi as envi
import spectral
from spectral import *
import matplotlib.pyplot as plt
import numpy as np
import config
import os
from PIL import Image
#from torch.utils.data import Dataset, DataLoader
#from torchvision.utils import save_image
import random
from collections import defaultdict
#import torch
#import spectral as spy
import scipy.io
from spectral.utilities.python23 import typecode, tobytes, frombytes
'''def draw(probs,val):
        csum = 0
        for i, p in enumerate(probs):
                csum  += p
                if csum > val:
                        return i


def draw_sample(prob,prob_val,noise_size=100):
        max_samples=noise_size
        counter = defaultdict(int)
        for i in range(max_samples):
                choice = draw(prob,random.choice(prob_val))
                counter[choice] += 1

        vals = tf.convert_to_tensor(list(counter.values()))
        return vals'''

def parse_mat(o):
        # this seems to happen for lists (1D cell arrays usually)
        if o.__class__ == np.ndarray and o.dtype == np.object and len(o.shape) > 0:
                assert len(o.shape) == 1, "didn't see this coming"
                return [parse_mat(entry) for entry in o]

        # this would be a matlab struct
        if o.__class__ == scipy.io.matlab.mio5_params.mat_struct:
                return {fn: parse_mat(getattr(o, fn)) for fn in o._fieldnames}

        # this means this should either be a regular numeric matrix
        # or a scalar
        return o

class HsiDataset():
    def __init__(self, root_dir):
        self.root_dir=root_dir


    def getitem(self, noise_size):
        mat=scipy.io.loadmat('Indian_pines.mat')
        keys = [k for k in mat.keys() if not k.startswith('_')]

        if len(keys)==1:
            img=parse_mat(mat[keys[0]])
        else:
            img=[parse_mat(mat[k]) for k in keys]
        print('keys are following: ')
        print(keys)
        print(np.min(img))
        print(np.max(img))
        data = 2*np.array((img - np.min(img))/np.ptp(img))-1
        print(np.max(data))
        data = np.float32(data)
    
        return data