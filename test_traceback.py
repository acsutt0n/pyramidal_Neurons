import sys, traceback
from pyramidal_readExportedGeometry import *

def do(hocfile):
  geo = demoRead(hocfile)


if __name__ == '__main__':
  hocfile = sys.argv[1]
  try:
    do(hocfile)
    with open('output.txt', 'w') as ofile:
      ofile.write('True')
  except:
    print('Logging traceback')
    traceback.print_exc(file=open('output.txt', 'w'))
    sys.exit(1)
    
