# https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/NSynth.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

fname = os.path.join(os.environ['INPUT_AUDIO_PATH'], 'horn.wav')
print(fname)

sr = 16000
audio = utils.load_audio(fname, sample_length=40000, sr=sr)
sample_length = audio.shape[0]
print('{} samples, {} seconds'.format(sample_length, sample_length / float(sr)))

CHECKPOINT_PATH = os.environ['WAVENET_CKPT_PATH']
encoding = fastgen.encode(audio, CHECKPOINT_PATH, sample_length)
print(encoding.shape)

# Synthesize back into output audio.
fastgen.synthesize(encoding, save_paths=['generated.wav'],
  samples_per_save=sample_length,
  checkpoint_path=CHECKPOINT_PATH)
