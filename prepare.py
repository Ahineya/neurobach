import sys
import argparse
import pretty_midi
import numpy
import json

CONST_PIANO_CHANNEL = 0

def prepare():
  parser = argparse.ArgumentParser()
  parser.add_argument('file', action='store')
  args = parser.parse_args(sys.argv[2:])

  piano_roll = midi_to_piano_roll(args.file)
  dict_time_notes = piano_roll_to_time_dict(piano_roll)
  song = dict_time_notes_to_text(dict_time_notes)

  print(song)

def midi_to_piano_roll(file):
  midi_pretty_format = pretty_midi.PrettyMIDI(file)
  piano_midi = midi_pretty_format.instruments[CONST_PIANO_CHANNEL]
  return piano_midi.get_piano_roll(fs=5)

def piano_roll_to_time_dict(sample):
  times = numpy.unique(numpy.where(sample > 0)[1])
  index = numpy.where(sample > 0)
  dict_time_notes = {}

  for time in times:
    index_where = numpy.where(index[1] == time)
    notes = index[0][index_where]
    dict_time_notes[time.item()] = notes.tolist()
  
  return dict_time_notes

def dict_time_notes_to_text(dict_time_notes):
  max_key = max(song, key=int)

  song_list = []

  for i in range(max_key + 1):
    if i in song:
      notes = 'and'.join(str(note) for note in song[i])
      song_list.append(notes)
    else:
      song_list.append('e')

  return ' '.join(song_list)

prepare()