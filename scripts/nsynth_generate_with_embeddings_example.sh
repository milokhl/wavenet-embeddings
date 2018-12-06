# Generate audio samples with precomputed embedddings (i.e from interpolation).
# TODO: where should embeddings be?

nsynth_generate \
--checkpoint_path=${WAVENET_CKPT_PATH} \
--source_path=${INPUT_AUDIO_PATH} \
--save_path=${GEN_AUDIO_PATH}/examples/ \
--encodings=true \
--batch_size=4
