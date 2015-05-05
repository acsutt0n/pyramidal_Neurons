

dir1 = '/home/acs/Data/vanhooser/RIgny/'
dir2 = '/home/acs/Data/vanhooser/RIgny2/'
dir3 = '/home/acs/Data/vanhooser/RIuxc/'
dir4 = '/home/acs/Data/vanhooser/RIxpc/'
dir5 = '/home/acs/Data/vanhooser/RIxpc2/'
dir6 = '/home/acs/Data/vanhooser/RIydy/'
dir7 = '/home/acs/Data/vanhooser/RIydy2/'

dirs = [dir1, dir2, dir3, dir4, dir5, dir6, dir7]

root1 = 'RIgny_'
root2 = 'RIgny2_'
root3 = 'RIuxc_'
root4 = 'RIxpc_'
root5 = 'RIxpc2_'
root6 = 'RIydy_'
root7 = 'RIydy2_'

roots = [root1, root2, root3, root4, root5, root6, root7]

for d in range(len(dirs)):
  hocs = get_hocfiles(dirs[d])
  props = get_properties(hocs, roots[d]+'.txt')
  save_dict(props, roots[d])

