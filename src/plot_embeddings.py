import os
import numpy as np
from matplotlib import pyplot as plt

def PlotModifiedVsRaw(raw_encoding, modified_encoding):
  """
  Show components of embeddings over time.
  """
  fig, axs = plt.subplots(2, 1, figsize=(10, 7))
  axs[0].plot(np.squeeze(raw_encoding));
  axs[0].set_title('Raw Encoding')
  axs[1].plot(np.squeeze(modified_encoding));
  axs[1].set_title('Modified Encoding')
  plt.show()

def PlotMultiple(encodings, titles):
  num_plots = len(encodings)
  fig, axs = plt.subplots(num_plots, 1, figsize=(10, 7))

  for i in range(num_plots):
    axs[i].plot(np.squeeze(encodings[i]))
    axs[i].set_title(titles[i])

  plt.show()

if __name__ == '__main__':
  embeddings_folder = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/raw_embeddings/'
  
  raw_fname = os.path.join(embeddings_folder, 'Alesis-Fusion-English-Horn-C5.npy')
  modified_fname = os.path.join(embeddings_folder, 'Alesis-Fusion-English-Horn-C5.npy')

  raw = np.load(raw_fname)
  modified = np.load(modified_fname)

  PlotModifiedVsRaw(raw, modified)
