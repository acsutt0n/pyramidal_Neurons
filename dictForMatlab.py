# dictForMatlab.py
# writes matlab-friendly data output

def save_lol(lol, fname):
  # save a list of lists (array) into a file with csv or space sep var
  with open(fname, 'wb') as fOut:
    for l in lol:
      for i in l:
        fOut.write(str(i))
        fOut.write(' ') # here is delimiter
      fOut.write('\n')
  return


def save_dict(dic, root_name):
  for k in dic.keys():
    save_lol(dic[k], root_name+k+'.txt')
