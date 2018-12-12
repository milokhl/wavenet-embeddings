import os
from random import random
import numpy as np
from utils import GetFilesInDir
from matplotlib import pyplot as plt

# https://www.audiocheck.net/audiofrequencysignalgenerator_sinetone.php

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/08_components_vs_pitch'

  encoding_folder = os.path.join(base_dir, 'embeddings')

  # Sort files in MIDI note order.
  files = GetFilesInDir(encoding_folder, full_path=False)
  files.sort()

  pitches = []
  component_values = [[] for _ in range(16)]

  for f in files:
    p = f
    # p = p[p.find('-')+1:]
    # p = p[:p.find('-')]
    p = p[p.find('sin_')+4:]
    p = p[:p.find('Hz')]

    pitches.append(int(p))
    encoding = np.squeeze(np.load(os.path.join(encoding_folder, f)))

    for i in range(16):
      component_values[i].append(np.mean(encoding[:,i]))

  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
  for i in range(len(component_values)):
    plt.plot(pitches, component_values[i], 'x', color=colors[(i+1) % 8])

  plt.title('Average Component Magnitude vs. Frequency')
  plt.ylabel('Magnitude')
  plt.xlabel('Frequency (Hz)')
  plt.show()
