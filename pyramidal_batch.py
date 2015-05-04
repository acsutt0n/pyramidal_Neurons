# pyramidal_batch.py - eliminate redundant segments for an entire directory
# usage: python pyramidal_batch.py directory(str)

import sys, os
import json
from pyramidal_traceback import *
from XmlToHoc_simple import *
from neuron_getProperties import *



def get_filelist(directory):
  all_files = os.listdir(directory)
  xml_files = [i for i in all_files if i.split('.')[1]=='xml']
  fnames = [i.split('.')[0] for i in xml_files]
  hoc_files = [i+'.hoc' for i in fnames]
  if len(xml_files) == len(hoc_files):
    return xml_files, hoc_files
  else:
    print('xml and hoc files do not match')
    return False


def run_batch(directory):
  xml_files, hoc_files = get_filelist(directory)
  xml_files = [directory+i for i in xml_files]
  hoc_files = [directory+i for i in hoc_files]
  success = []
  # first convert xml to hoc
  for f in range(len(xml_files)):
    SkelHoc(xml_files[f], hoc_files[f])
  # now process redundant sections for each hoc file
  for f in hoc_files:
    try:
      remove_redundant(f)
      success.append(f)
    except:
      print('Could not remove for %s' %f)
  return success 


### continue to analysis part?
def get_properties(success, save=None):
  geo_files = []
  for f in success:
    try:
      geo_files.append(demoRead(f))
    except:
      pass
  branchAngles = [branch_angles(g) for g in geo_files]
  pathLengths = [path_lengths2(g)[0] for g in geo_files]
  pathTorts = [path_lengths2(g)[1] for g in geo_files]
  shollDistances = [interpolate_nodes(g) for g in geo_files]
  asymmetryLengths = [tips_asymmetry(g)[0] for g in geo_files]
  asymmetryTips = [tips_asymmetry(g)[1] for g in geo_files]
  simple_asymmetryLengths = [simplify_asymmetry(x,y)[0] for x, y in
                             zip(asymmetryLengths, asymmetryTips)]
  simple_asymmetryTips = [simplify_asymmetry(x,y)[1] for x,y in
                             zip(asymmetryLengths, asymmetryTips)]
  props  =  {'fileNames': success,
             'branchAngles': branchAngles,
             'pathLengths': pathLengths,
             'pathTortuosities': pathTorts,
             'shollDistances': shollDistances,
             'asymmetryLengths': asymmetryLengths,
             # 'asymmetryTips': asymmetryTips,
             'simple_asymmetryLengths': simple_asymmetryLengths,
             'simple_asymmetryTips': simple_asymmetryTips}
  if save:
    savefile = open(save, 'w')
    json.dump(props, savefile)
  return props



if __name__ == '__main__':
  args = sys.argv
  run_batch(args[1])
