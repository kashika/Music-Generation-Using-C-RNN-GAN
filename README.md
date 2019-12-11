# Music-Generation-Using-C-RNN-GAN

This project is generating of good quality music by understanding the patterns from the existing music data. As a part of the solution 
I have explored different methodologies for music generation including Many-to-Many RNN and C-RNN-GAN. I have also evaluate and compared
the results from implementation of different methodologies and finally demo the sample music.

Packages Required:
Keras
TensorFlow
PyTorch
Sklearn
Pandas
Seaborn
Matplotlib
Numpy
Scipy
midi

Dataset:
1. RNN and LSTM
405 Folk Music Tunes converted in ABC Notation 
Sequence of texts
http://abc.sourceforge.net/NMD/

2. C-RNN-GAN
3697 music files in midi format of classical music
The data is normalized to a tick resolution of 384 per quarter note
https://github.com/olofmogren/c-rnn-gan


Training data was collected from the web in the form of music files in midi format, containing well-known works of classical music.
Each midi event of the type “note on” was loaded and saved together with its duration, tone, intensity (velocity), and time since 
beginning of last tone. The tone data is internally represented with the corresponding sound frequency. Internally, all data is 
normalized to a tick resolution of 384 per quarter note. The data contains 3697 midi files from 160 different composers of classical 
music. 

Part-1 represents meta data. Lines in the Part-1 of the tune notation, beginning with a letter followed by a colon, indicate various 
aspects of the tune such as the index, when there are more than one tune in a file (X:), the title (T:), the time signature (M:), the 
default note length (L:), the type of tune (R:) and the key (K:).
Part-2 represents the tune, which is a sequence of characters where each character represents some musical note.

MIDI itself does not make sound, it is just a series of messages like “note on,” “note off,” “note/pitch,” “pitch-bend,” and many more.
These messages are interpreted by a MIDI instrument to produce sound. A MIDI instrument can be a piece of hardware (electronic keyboard, 
synthesizer) or part of a software environment (ableton, garageband, digital performer, logic


Train:

python train.py

