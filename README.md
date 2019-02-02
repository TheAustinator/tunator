# tunator

Note: I have not had time to organize this repo yet, but almost all code is contained in lstm.py. If you're interested discussing or working on the repo, I'd be happy to set up a call with you (austinmckay303@gmail.com).

## Usage
Will fill in later, but contact me if you need help in the meantime.

## Scope
The goal of this project is to allow users to synthesize melodies in the same musical style as their favorite singers. There are three steps in this process:

Step 1: Extract vocals from raw audio

Step 2: Transcribe vocals to MIDI format

Step 3: Generate new, similar MIDIs

All steps are a work in progress based on prior work.

## Step 1: Vocal Isolation
There are several implementations based on (Spotify's U-Net)[https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf] on Github, which I haven't had time to look through yet.

## Step 2: MIDI Transcription
It appears that  (aniawsz's)[https://github.com/aniawsz] (Spectral_Analyzer)[https://github.com/aniawsz/rtmonoaudio2midi/blob/master/audiostream.py] could be easily adapted to this purpose.

## Step 3: Melody Generation
I would like to implement something based on magenta's (MusicVAE)[https://arxiv.org/pdf/1803.05428.pdf]. Most of the work on this repo so far has been building the pipeline. Currently, the pipeline from MIDI files to generated MIDI files is complete, however, the current model is just a basic LSTM as a placeholder. 

## Ideas and Challenges

One preprocessing step that could make the modeling much easier is a model that can parition the input MIDI files into parts (e.g. verse, chorus, bridge). Since each of these sequences tend to sound different, classifying them beforehand could allow users to sample one type of melody specifically. 
