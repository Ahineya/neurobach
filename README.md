# Neurobach

Neural network for training on Midi keyboard pieces

Most of the ideas and code is described here https://towardsdatascience.com/generate-piano-instrumental-music-by-using-deep-learning-80ac35cdbd2e

The neural network layer though is not the same as in the original article. I am using textgenrnn to train Tensorflow, treating Midi data as a word sequences.

Drop me a message on Reddit `u/Ahineya_it`, if you have any questions or ideas. Also ping me to fix the grammar mistakes :)

### Setup

Use virtualenv, and install required modules:

```
$ pip install -r requirements.txt
```

### Usage

#### Testing current model

I have included the model I've trained — it is called "bach". To test it simply execute:

```
$ python3 generate.py
```

Midi file will be generated in the `generated` directory.

#### Training new model

To train a new model, you should execute next steps:

##### Midi preparation

Place all your midi files in the `midi` directory (currently it contains all Bach pieces, you can use those for testing the training process). After that execute:

```
$ ./prepare-midi.sh 
```

This command will go through all midi files and convert them to a
specific text format that can be understood by textgenrnn. It will create the `output/output.txt` file. Some files will generate an
error — it seems like not all Midi files are completely valid.

Also, it is better to have all the midi pieces transposed to the same key.

##### Training configuration

Training configuration is located in a `trainer.py` script. Currently
there is a configuration I have used for training the network.
Tune parameters as you think they should be — there are comments
in a file that will help to determine, what should be done.

##### Training

Run this command:

```
$ ./train.sh YOUR_MODEL_NAME
```

Replace __YOUR_MODEL_NAME__ with your model name. Depending on a configuration, training will take some time. On Macbook Pro 16' it took me about 3-4 hours to fully train the network.

##### Generating new pieces

Use this command to generate a new pieces with your trained model:

```
$ python3 generate.py --model-name YOUT_MODEL_NAME --temperature TEMPERATURE
```

Replace __YOUR_MODEL_NAME__ with the model name you have used during the training phase. Replace __TEMPERATURE__ with a float value from 0 to somewhere around 3. 1.5 is fine for the 'bach' model.

A new piece will be generated in the `generated` directory.

Pay attention, that all generated pieces are quantized to a white piano keys. You can take a look in `bach.py` file to turn it off if you don't need it, or to create a new scale for quantizing.
