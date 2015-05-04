# pyramidal_parser.py - figure out whether the program ran correctly
# usage: python pyramidal_parser.py output.txt

import sys


def parse_it(output):
  with open(output, 'r') as Out:
    for line in Out:
      if line:
        splitLine = line.split(None)
        if splitLine[0] == 'Number': # this signals a successful run
          print("True True")
          return
          # it worked
        elif splitLine[0] == 'Traceback':
          print(get_segments(Out))
          return


def get_segments(fObject):
  Lines = []
  for line in fObject:
    Lines.append(line)
  for l in Lines:
    if l.split(None)[0] == 'AssertionError:':
      seg0 = int(l.split(None)[4].split('[')[1].split(']')[0])
      seg1 = int(l.split(None)[6].split('[')[1].split(']')[0])
      print('%i %i ' %(seg0, seg1))
      return
  print('False False')
  return
  

##
if __name__ == '__main__':
  args = sys.argv
  outputfile = args[1]
  parse_it(outputfile)
