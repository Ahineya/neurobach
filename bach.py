from textgenrnn import textgenrnn
import random
import json
import pretty_midi
import numpy as np
import uuid

MAX_PIECE_LENGTH = 1000
cmaj = [21, 23, 24, 26, 28, 29, 31, 33, 35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96, 98, 100, 101, 103, 105, 107, 108]

# Quantizes the note to the closest one in scale
def quantize(note, scale):
  if note in scale:
    return str(note)
  else:
    left = 0
    right = 0

    for n in scale:
      if n < note:
        left = n
      else:
        right = n
        break
    
    min_diff = note - left
    max_diff = right - note

    if min_diff < max_diff:
      return str(left)
    
    if max_diff < min_diff:
      return str(right)

    if random.choice([True, False]):
      return str(left)
    else:
      return str(right)

def piano_roll_to_pretty_midi(piano_roll, fs=5, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def generated_to_json(text):
  notes = text.split(' ')
  parsed_notes = []
  for note in notes:
    if note == 'e':
      parsed_notes.append(note)
    else:
      split_notes = note.split('and')
      notes_list = list(map(lambda n: quantize(int(n), cmaj), split_notes))
      parsed_notes.append(','.join(notes_list))
  
  return parsed_notes

def json_to_midi(notes):
  array_piano_roll = np.zeros((128,MAX_PIECE_LENGTH), dtype=np.int16)

  for i, note in enumerate(notes):
    if note == 'e':
        pass
    else:
        splitted_note = note.split(',')
        for j in splitted_note:
            array_piano_roll[int(j),i] = 1
  
  generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll,5)
  print("Tempo {}".format(generate_to_midi.estimate_tempo()))
  for note in generate_to_midi.instruments[0].notes:
    note.velocity = 100
  filename = str(uuid.uuid1())
  filename = filename + '.mid'
  generate_to_midi.write('generated/' + filename)
  return filename

def generate(model_name, temperature):
  generator = textgenrnn(weights_path = model_name + '_weights.hdf5', vocab_path = model_name + '_vocab.json', config_path = model_name + '_config.json')
  for text in generator.generate(1, return_as_list=True, temperature=temperature):
    filename = json_to_midi(generated_to_json(text))
  return filename
