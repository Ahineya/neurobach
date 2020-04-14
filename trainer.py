import sys
import argparse

from textgenrnn import textgenrnn

def train():
  parser = argparse.ArgumentParser()
  parser.add_argument('file', action='store')
  parser.add_argument('--model-name', action='store', required=True)
  parser.add_argument('--epochs', action='store', default=25, type=int)

  args = parser.parse_args(sys.argv[2:])

  model_cfg = {
    # set to True if want to train a word-level model (requires more data and smaller max_length)
    'word_level': True,
    # number of LSTM cells of each layer (128/256 recommended)
    'rnn_size': 256,
    # number of LSTM layers (>=2 recommended)
    'rnn_layers': 4,
    # consider text both forwards and backward, can give a training boost
    'rnn_bidirectional': False,
    # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_length': 10,
    # maximum number of words to model; the rest will be ignored (word-level model only)
    'max_words': 10000
  }

  train_cfg = {
    # set to True if each text has its own line in the source file
    'line_delimited': True,
    # set higher to train the model for longer
    'num_epochs': args.epochs,
    # generates sample text from model after given number of epochs
    'gen_epochs': 1,
    # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'train_size': 0.9,
    # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'dropout': 0.1,
    # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'validation': False,
    # set to True if file is a CSV exported from Excel/BigQuery/pandas
    'is_csv': False
  }

  textgen = textgenrnn(name=args.model_name)

  train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file

  train_function(
    file_path=args.file,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=1024,
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=100,
    word_level=model_cfg['word_level']
  )

def generate():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-name', action='store')
  parser.add_argument('--temperature', action='store', default=0.2, type=float)
  parser.add_argument('--n', action='store', default=10, type=int)

  args = parser.parse_args(sys.argv[2:])

  generator = textgenrnn(weights_path=args.model_name + '_weights.hdf5',vocab_path=args.model_name + '_vocab.json',config_path=args.model_name + '_config.json')

  for text in generator.generate(args.n, return_as_list=True, temperature=args.temperature):
    print(text)

parser = argparse.ArgumentParser()
parser.add_argument('command', action='store')

args = parser.parse_args(sys.argv[1:2])

if(args.command == 'train'):
  train()
elif(args.command == 'generate'):
  generate()
