# https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/NSynth.ipynb

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

CHECKPOINT_PATH = os.environ['WAVENET_CKPT_PATH']
fname = os.path.join(os.environ['INPUT_AUDIO_PATH'], 'horn.wav')
print('INFO] Input audio path:', fname)

sr = 16000
audio = utils.load_audio(fname, sample_length=40000, sr=sr)
sample_length = audio.shape[0]
print('{} samples, {} seconds'.format(sample_length, sample_length / float(sr)))

# Extract temporal embeddings from input audio.
t0 = time.time()
encoding = fastgen.encode(audio, CHECKPOINT_PATH, sample_length)
print('Encoding shape:', encoding.shape)
print('Generated embeddings in %f sec' % (time.time() - t0))

# Synthesize back into output audio.
t0 = time.time()
fastgen.synthesize(encoding, save_paths=['generated.wav'],
  samples_per_save=sample_length,
  checkpoint_path=CHECKPOINT_PATH)
print('Generated output audio in %f sec' % (time.time() - t0))
