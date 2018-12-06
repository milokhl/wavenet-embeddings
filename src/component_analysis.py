import os
from utils import LoadEmbeddingsFromDir, SafeMakeDir

# This will be used to save the output.
EXPERIMENT_NAME = '01-component-analysis'

def RunExperiment(options):
  GEN_EMBEDDING_PATH = os.environ['GEN_EMBEDDING_PATH']

  SafeMakeDir(os.path.join(options[]))

if __name__ == '__main__':
  options = {}
  RunExperiment(options)
