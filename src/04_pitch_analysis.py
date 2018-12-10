import os, time
import numpy as np
import tensorflow as tf

from utils import GetFilesInDir, SafeMakeDir
from generate_embeddings import GenerateEmbeddings

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/04_pitch_analysis'
  input_folder = os.path.join(base_dir, 'audio/')
  output_folder = os.path.join(base_dir, 'embeddings/')
  ckpt_path = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/data/wavenet-ckpt/model.ckpt-200000'

  # Make sure output folder exists.
  SafeMakeDir(output_folder)

  # Generate embeddings for each audio clip.
  t0 = time.time()
  GenerateEmbeddings(input_folder, output_folder, ckpt_path)
  print('Done (%f sec).' % (time.time() - t0))
