# Final Project: WaveNet Embedding Analysis

This is my Final project for 21M.080, Introduction to Music Technology.

## Motivation

The key idea behind an autoencoder network is that the learned embedding can represent a complex sound in a small number of parameters (16 in the case of NSynth WaveNet). However, if we think of each component of these embeddings as a knob on a synth, we don’t really know what each one does (at least I don't). My goal for this project is to investigate how numerical alterations to the embeddings affects qualitative aspects of the sound. Is there an interpretable meaning to each of the 16 parameters?

## References

- [NSynth Model on Github](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth)
- [Neural Audio Synthesis of Music Notes with WaveNet Autoencoders](https://arxiv.org/pdf/1704.01279.pdf)
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)
