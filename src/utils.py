import numpy as np
from os import listdir
from os.path import isfile, join, isdir, exists

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
  """
  files = [f for f in listdir(path) if isfile(join(path, f))]

  embeddings = []
  for f in files:
    embeddings.append(np.load(join(path, f)))

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
