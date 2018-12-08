import os
import numpy as np
from utils import GetFilesInDir, SafeMakeDir

def MakeZeroComponentEmbeddings(input_folder, output_folder):
  raw_embeddings = GetFilesInDir(input_folder, full_path=True)

  for f in raw_embeddings:
    head, tail = os.path.split(f)

    encoding = np.squeeze(np.load(f)) # Remove empty first dimension.
    num_samples = encoding.shape[0]

    # For each component, set to zero and save the modified array.
    for i in range(16):
      encoding_cp = np.copy(encoding)
      encoding_cp[:,i] = 0

      # Save a modified array for this component.
      outfile = os.path.join(output_folder, 'zeros_%d_%s' % (i, tail))
      np.save(outfile, encoding_cp)

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/01_component_analysis'
  input_folder = os.path.join(base_dir, 'raw_embeddings/')
  output_folder = os.path.join(base_dir, 'modified_embeddings/')

  # Make sure output folder exists.
  SafeMakeDir(output_folder)

  MakeZeroComponentEmbeddings(input_folder, output_folder)
