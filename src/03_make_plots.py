import os
import numpy as np
from plot_embeddings import PlotModifiedVsRaw

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/03_sign_analysis'

  raw_folder = os.path.join(base_dir, 'raw_embeddings')
  mod_folder = os.path.join(base_dir, 'modified_embeddings')
  raw_fname = os.path.join(raw_folder, 'Alesis-Fusion-English-Horn-C5.npy')
  mod_fname = os.path.join(mod_folder, 'gain2x_0_Alesis-Fusion-English-Horn-C5.npy')

  raw = np.load(raw_fname)
  modified = np.load(mod_fname)

  PlotModifiedVsRaw(raw, modified)
