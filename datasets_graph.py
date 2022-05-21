# dataloader for graphdata
# Author young-hoon Ji
# last update 220511

import os
import torch
import numpy as np
import csv
from glob import glob
import cv2

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

# torch tensor input
def show_graph_data(x, node_range=(0,584)):
  max_node_num, num_coord = x.shape
  x_np = x.detach().cpu().numpy()

  img = np.zeros((max(node_range), max(node_range)), dtype=np.uint8)

  for i in range(max_node_num):
    node_y, node_x = int(x_np[i][0]*max(node_range)), int(x_np[i][1]*max(node_range))
    img[node_y, node_x] = 255

  return img

class VesselNodeDataLoader(torch.utils.data.Dataset):
  def __init__(self, data_path):
    self.data_path = data_path
    self.max_node_num = 0
    self.node_coord_range = (0, 584)
    self.graph_data = None
    self.graph_data_length = []

    self.graph_data = self.data_load()

  def __len__(self):
    return self.graph_data.shape[0]

  def __getitem__(self, idx):
    data = self.graph_data[idx]
    length = self.graph_data_length[idx] # added for dataset part
      
    return data, length

  def data_load(self):
    data = np.load(self.data_path, allow_pickle=True)
    self.total_num_data = data.shape[0]
    self.coord_num = 2 # x, y coord
    # searching max_node data to add padding
    for i in range(self.total_num_data):
      node_num = data[i].shape[0]
      self.graph_data_length.append(node_num)
      if(self.max_node_num < node_num):
        self.max_node_num = node_num

    padded_dataset = self.data_padding(data)

    # data normalization
    padded_dataset_norm = self.data_normalize(padded_dataset)

    return padded_dataset_norm

  def data_normalize(self, x):
    # normalize data to [0,1] scale
    x = x.copy().astype(np.float32)
    x = x / max(self.node_coord_range)

    return x

  def data_padding(self, x):
    padded_dataset = np.zeros((self.total_num_data, self.max_node_num, self.coord_num), dtype=np.int64)

    graph_data_length_npy = np.array(self.graph_data_length)
    graph_length_sort = np.argsort(graph_data_length_npy)[::-1]
    for idx, j in enumerate(graph_length_sort):
      pad_size = self.max_node_num - x[j].shape[0]
      padded_data = np.pad(x[j], ((0,pad_size),(0,0)), "constant", constant_values=0)
      padded_dataset[idx] = padded_data

    self.graph_data_length.sort(reverse=True)

    return padded_dataset

  def data_flip(self, x):
    return None

  def data_shuffle(self, x):
    np.take(x,np.random.permutation(x.shape[0]),axis=0,out=x)
    return None

  
if __name__=="__main__":
  path = "./utils_yhji/node_npy_data/220511_sampling12/220511_sampling12.npy"
  batch_size = 4
  num_worker = 1
  vessel_node_dataset = VesselNodeDataLoader(path)
  test_dataset_loader = torch.utils.data.DataLoader(vessel_node_dataset, batch_size=4, num_workers=num_worker, shuffle=True)


  print("debug")
  temp = next(iter(test_dataset_loader))
  img = show_graph_data(temp[0])
  print("debug")





  # 
  # temp_npy = temp.numpy()
  # debug_path = "./output/debug/220513_dataloader/"
  # for i in range(4):
  #   viz_map = np.zeros((584, 584), dtype=np.uint8)
  #   for j in range(623):
  #     node_y, node_x = int(temp_npy[i][j][0]*584), int(temp_npy[i][j][1]*584)
  #     viz_map[node_y, node_x] = 255

  #   cv2.imwrite(debug_path+str(i)+".png", viz_map)