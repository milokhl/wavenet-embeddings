import os, time
import numpy as np
import tensorflow as tf

from utils import GetFilesInDir, SafeMakeDir
from generate_embeddings import GenerateEmbeddings
from synthesize_audio import SynthesizeAudio

if __name__ == '__main__':
  base_dir = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/06_pitch_comparison'
  input_folder = os.path.join(base_dir, 'audio/')
  output_folder = os.path.join(base_dir, 'embeddings/')
  ckpt_path = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/data/wavenet-ckpt/model.ckpt-200000'

  # Make sure output folder exists.
  SafeMakeDir(output_folder)

  # Generate embeddings for each audio clip.
  t0 = time.time()
  # GenerateEmbeddings(input_folder, output_folder, ckpt_path)
  print('Done (%f sec).' % (time.time() - t0))

  # Regenerate the audio to make sure that pitches are reproduced.
  tf.logging.set_verbosity(tf.logging.INFO)

  synth_input_folder = output_folder
  synth_output_folder = os.path.join(base_dir, 'synth_audio/')
  SafeMakeDir(synth_output_folder)

  t0 = time.time()
  SynthesizeAudio(synth_input_folder, synth_output_folder, ckpt_path)
  print('Done (%f sec).' % (time.time()-t0))
