import os
import numpy as np
from plot_embeddings import PlotMultiple
from utils import GetFilesInDir

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/06_pitch_comparison'
  encoding_folder = os.path.join(base_dir, 'embeddings')

  encodings = []
  titles = []

  files = GetFilesInDir(encoding_folder, full_path=True)
  for f in files:
    head, tail = os.path.split(f)
    encodings.append(np.load(f))
    titles.append(tail)

  # Need to sort by filename.
  tup_sorted = sorted(zip(titles, encodings))
  encodings = [el[1] for el in tup_sorted]
  titles = [el[0] for el in tup_sorted]

  PlotMultiple(encodings, titles)
