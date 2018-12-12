import os
import numpy as np
from plot_embeddings import PlotModifiedVsRaw

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/01_component_analysis'

  raw_folder = os.path.join(base_dir, 'raw_embeddings')
  mod_folder = os.path.join(base_dir, 'modified_embeddings')
  raw_fname = os.path.join(raw_folder, 'banjo.npy')
  mod_fname = os.path.join(mod_folder, 'zeros_0_banjo.npy')

  raw = np.load(raw_fname)
  modified = np.load(mod_fname)

  PlotModifiedVsRaw(raw, modified)
