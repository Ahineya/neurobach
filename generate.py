import bach
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', default='bach')
parser.add_argument('--temperature', action='store', default=1.5, type=float)
args = parser.parse_args(sys.argv[1:2])

model_name = args.model
temperature = args.temperature

print('Model: ' + model_name)
print('Temperature: ' + str(temperature))

bach.generate(model_name, temperature)