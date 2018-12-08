import os, time
import tensorflow as tf
import numpy as np
from magenta.models.nsynth.wavenet import fastgen
from magenta.models.nsynth.wavenet.fastgen import load_batch
from utils import LoadEmbeddingsFromDir, GetFilesInDir, load_batch_encodings

def SynthesizeAudio(input_folder, output_folder, ckpt_path):
  """
  Reconstruct audio from temporal embeddings.
  """
  # Define constants here.
  MAX_AUDIO_LENGTH_SECONDS = 2.0

  print('[INFO] Input folder:', input_folder)
  print('[INFO] Output folder:', output_folder)

  # Load in .npy embeddings as a batch array (batch_size, num_encodings, 16).
  npy_files = GetFilesInDir(input_folder, full_path=True)

  # Max sample length of 125 encodings * 32 ms = 4 seconds.
  max_encoding_length = int(1000.0 * MAX_AUDIO_LENGTH_SECONDS / 32.0)

  # Note: this gets renamed to load_batch_encodings in master.
  encodings_batch = load_batch_encodings(npy_files, sample_length=max_encoding_length)
  print('[INFO] Loaded %d embeddings.' % encodings_batch.shape[0])

  sr = 16000
  samples_per_encoding = sr / 1000 * 32 # 32 ms / encoding

  # Make an ordered list of output filenames.
  save_paths = []
  for f in npy_files:
    head, tail = os.path.split(f)
    outfile = 'synth_' + str(tail).replace('.npy', '.wav')
    outpath = os.path.join(output_folder, outfile)
    save_paths.append(outpath)

  # Use the decoder network to reconstruct audio from embeddings.
  # Doing this in batches is much faster.
  print('[INFO] Synthesizing output audio with %d samples (%f sec)' % \
        (max_encoding_length, MAX_AUDIO_LENGTH_SECONDS))
  
  t0 = time.time()
  fastgen.synthesize(encodings_batch,
                     save_paths=save_paths,
                     samples_per_save=5000,
                     checkpoint_path=ckpt_path)

  print('Generated output audio in %f sec' % (time.time()-t0))

if __name__ == '__main__':
  # Prints out the number of samples generated. Looks like ~100 samples per second.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Set up input and output directories.
  input_folder = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/raw_embeddings/'
  output_folder = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/audio/from_raw_embeddings/'
  ckpt_path = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/data/wavenet-ckpt/model.ckpt-200000'

  t0 = time.time()
  SynthesizeAudio(input_folder, output_folder, ckpt_path)
  print('Done (%f sec).' % (time.time()-t0))
