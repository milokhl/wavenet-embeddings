import os
import numpy as np
from plot_embeddings import PlotModifiedVsRaw
from utils import GetFilesInDir

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/07_vibrato_analysis'

  raw_folder = os.path.join(base_dir, 'raw_embeddings')
  mod_folder = os.path.join(base_dir, 'modified_embeddings')

  raw_file = os.path.join(raw_folder, 'Alesis-Fusion-English-Horn-C5.npy')
  mod_file = os.path.join(mod_folder, 'vibrato_Alesis-Fusion-English-Horn-C5.npy')

  raw = np.load(raw_file)
  mod = np.load(mod_file)
  PlotModifiedVsRaw(raw, mod)
