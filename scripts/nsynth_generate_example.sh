# Generate audio WITHOUT any modification to the embeddings.

nsynth_generate \
--checkpoint_path=${WAVENET_CKPT_PATH} \
--source_path=${INPUT_AUDIO_PATH} \
--save_path=${GEN_AUDIO_PATH}/examples/ \
--batch_size=1
