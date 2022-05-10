from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets

import cv2

DEBUG_PATH = "./output/debug/220510_1024/"
os.makedirs(DEBUG_PATH, exist_ok=True)

def image_grid(x):
  size = config.data.image_size
  channels = config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def show_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()

  # a = img.copy()
  # b = cv2.normalize(src=a, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
  # b = b.astype(np.uint8)
  # cv2.imwrite("./output/debug/test.png", b[:,:,::-1])

def save_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  #img = image_grid(x)

  img = cv2.normalize(src=x, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
  img = img.astype(np.uint8)

  #cv2.imwrite("./output/debug/test.png", img[:,:,::-1])
  max_save_iter = img.shape[0]
  for i in range(max_save_iter):
    cv2.imwrite(DEBUG_PATH+str(i)+".png", img[i][:,:,::-1])

####### cifar-10 generation code #######
# sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
# if sde.lower() == 'vesde':
#   from configs.ve import cifar10_ncsnpp_continuous as configs
#   ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
#   config = configs.get_config()  
#   sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
#   sampling_eps = 1e-5
# elif sde.lower() == 'vpsde':
#   from configs.vp import cifar10_ddpmpp_continuous as configs  
#   ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
#   config = configs.get_config()
#   sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
#   sampling_eps = 1e-3
# elif sde.lower() == 'subvpsde':
#   from configs.subvp import cifar10_ddpmpp_continuous as configs
#   ckpt_filename = "exp/subvp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
#   config = configs.get_config()
#   sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
#   sampling_eps = 1e-3
####### cifar-10 generation code #######

# Face Generation
# from configs.ve import ffhq_256_ncsnpp_continuous as configs
# ckpt_filename = "exp/ve/ffhq_256_ncsnpp_continuous/checkpoint_48.pth"
# config = configs.get_config()
# sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
# sampling_eps = 1e-5

from configs.ve import ffhq_ncsnpp_continuous as configs
ckpt_filename = "exp/ve/ffhq_ncsnpp_continuous/checkpoint_60.pth"
config = configs.get_config()
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sampling_eps = 1e-5


batch_size = 8 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

# @title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 # @param {"type": "number"}
n_steps =  1# @param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)

x, n = sampling_fn(score_model)
save_samples(x)
