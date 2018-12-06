# Sets up useful environment variables.

export WAVENET_PROJ_ROOT='/home/milo/mit/21M.080-music-tech/wavenet-embeddings'
export WAVENET_CKPT_PATH='/home/milo/mit/21M.080-music-tech/data/ckpt/model.ckpt-200000'

export GEN_EMBEDDING_PATH='${WAVENET_PROJ_ROOT}/generated/embeddings/'
export GEN_AUDIO_PATH='${WAVENET_PROJ_ROOT}/generated/audio'

export INPUT_DATA_PATH='/home/milo/mit/21M.080-music-tech/data'
export INPUT_AUDIO_PATH='${INPUT_DATA_PATH}/audio'
export INPUT_EMBEDDING_PATH='${INPUT_DATA_PATH}/embeddings'
