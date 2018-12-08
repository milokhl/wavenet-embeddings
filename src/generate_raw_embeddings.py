import os, time
import numpy as np
from magenta.models.nsynth.utils import load_audio
from magenta.models.nsynth.wavenet import fastgen
from utils import GetFilesInDir

def GenerateRawEmbeddings(input_folder, output_folder, ckpt_path):
  """
  Get all of the input audio files in data/audio/ and generated embeddings.
  """
  print('[INFO] Input folder:', input_folder)
  print('[INFO] Output folder:', output_folder)

  audio_files = GetFilesInDir(input_folder, full_path=True)
  print('[INFO] Found %d audio files.' % len(audio_files))

  sr = 16000 # Wavenet works at 16kHz.
  for i, f in enumerate(audio_files):
    t0 = time.time()
    head, tail = os.path.split(f)

    audio = load_audio(f, sample_length=40000, sr=sr)
    sample_length = audio.shape[0]

    print('[INFO] Processing file %s: %d samples, %f seconds' % \
          (tail, sample_length, sample_length / float(sr)))

    # Extract temporal embeddings from input audio.
    encoding = fastgen.encode(audio, ckpt_path, sample_length)

    # Save to the output folder.
    outfile = str(tail).replace('.wav', '.npy')
    np.save(os.path.join(output_folder, outfile), encoding)
    print('[INFO] Finished in %f sec.' % (time.time() - t0))

if __name__ == '__main__':
  input_folder = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/data/audio/'
  output_folder = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/generated/raw_embeddings/'
  ckpt_path = '/home/milo/mit/21M.080-music-tech/wavenet-embeddings/data/wavenet-ckpt/model.ckpt-200000'

  t0 = time.time()
  GenerateRawEmbeddings(input_folder, output_folder, ckpt_path)
  print('Done (%f sec).' % (time.time() - t0))
