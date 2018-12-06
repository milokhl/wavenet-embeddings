# Extracts embeddings from input audio and saves as .npy file.

nsynth_save_embeddings \
--checkpoint_path=${WAVENET_CKPT_PATH} \
--source_path=${INPUT_AUDIO_PATH} \
--save_path=${GEN_EMBEDDING_PATH}/examples/ \
--batch_size=4
