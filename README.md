# Final Project: WaveNet Embedding Analysis

This is my Final project for 21M.080, Introduction to Music Technology.

## Motivation

The key idea behind an autoencoder network is that the learned embedding can represent a complex sound in a small number of parameters (16 in the case of NSynth WaveNet). However, if we think of each component of these embeddings as a knob on a synth, we don’t really know what each one does (at least I don't). My goal for this project is to investigate how numerical alterations to the embeddings affects qualitative aspects of the sound. Is there an interpretable meaning to each of the 16 parameters?

## Setup
```
sudo apt-get install libasound2-dev libjack-dev # Needed for rtmidi package.
pip install magenta-gpu==0.3.12 # To work with tensorflow 1.10

cd wavenet-embeddings

# This is where output files are written.
mkdir generated/
```

You'll need to download pretrained weights for WaveNet from [here](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth). Put them in ```data/wavenet-ckpt/```.

I've included some of the raw audio samples that I used in ```data/audio/```.

## Running Experiments

Each python script in ```src/``` corresponds to a particular experiment that I ran. You should only have to modify the ```base_dir``` variable in each script to run this on your machine. Each experiment will also require you to create a corresponding output folder in ```generated/```. Create a folder within the experiment generate folder with the same name as ```input_folder``` and place the desired input data in it. Running the experiment script will automatically create the output folder and save results to it.

## References
- [NSynth Model on Github](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth)
- [Neural Audio Synthesis of Music Notes with WaveNet Autoencoders](https://arxiv.org/pdf/1704.01279.pdf)
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)
