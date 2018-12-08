import numpy as np
from os import listdir
from os.path import isfile, join, isdir, exists, split

def GetFilesInDir(path, full_path=True):
  """
  Get all filenames in directory. Ignores other directories.
  """
  if not full_path:
    return [f for f in listdir(path) if isfile(join(path, f))]
  else:
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

def LoadEmbeddingsFromDir(path):
  """
  Loads in all of the .npz embeddings from a specified directory.

  Returns: a list of tuples like (file_name.npy, np.ndarray)
  """
  files = GetFilesInDir(path, full_path=True)

  embeddings = []
  for f in files:
    head, tail = split(f)
    embeddings.append((tail, np.load(f)))

  return embeddings

def SafeMakeDir(path):
  """
  Checks if a path exists, and creates it if not.
  """
  if not exists(path):
    print('Path %s does not exist, creating' % path)
    os.path.mkdir(path)
  else:
    print('Path %s already exists, doing nothing' % path)

def load_batch_encodings(files, sample_length=125):
  """
  Link: https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/fastgen.py

  Load a batch of encodings from .npy files.
  Args:
    files: A list of filepaths to .npy files
    sample_length: Maximum sample length
  Raises:
    ValueError: .npy array has wrong dimensions.
  Returns:
    batch: A padded array encodings [batch, length, dims]
  """
  batch = []
  # Load the data
  for f in files:
    data = np.load(f)
    if data.ndim != 2:
      if data.shape[0] != 1:
        raise ValueError("Encoding file with 3 dims should have 1st dimension == 1")

      # Get rid of batch_size = 1 first dimension if present.
      data = np.squeeze(data)
    
    length, channels = data.shape
    
    # Add padding or crop if not equal to sample length
    if length < sample_length:
      padded = np.zeros([sample_length, channels])
      padded[:length, :] = data
      batch.append(padded)
    else:
      batch.append(data[:sample_length])

  # Return as an numpy array
  batch = np.array(batch)
  return batch
