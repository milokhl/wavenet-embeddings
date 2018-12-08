import os, time
import numpy as np
import tensorflow as tf

from utils import GetFilesInDir, SafeMakeDir
from synthesize_audio import SynthesizeAudio

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

  # Step 1: process all embeddings.
  # MakeZeroComponentEmbeddings(input_folder, output_folder)

  # Step 2: synthesize the modified embeddings back into audio.
  tf.logging.set_verbosity(tf.logging.INFO)

  synth_input_folder = output_folder
  synth_output_folder = os.path.join(base_dir, 'audio/')
  SafeMakeDir(synth_output_folder)
  ckpt_path = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/data/wavenet-ckpt/model.ckpt-200000'

  t0 = time.time()
  SynthesizeAudio(synth_input_folder, synth_output_folder, ckpt_path)
  print('Done (%f sec).' % (time.time()-t0))
