# retrieve simple properties from a geo instance

from pyramidal_readExportedGeometry import *
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# helper functions
def name(geo):
  return geo.fileName.split('/')[-1].split('.')[0]


def farthest_pt(pts):
  dmax = 0
  for i in pts:
    for j in pts:
      if dist3(i,j) > dmax:
        dmax = dist3(i,j)
  return dmax


def checko(obj):
  unique_files, unique_items, unique_cells = None, None, None
  if type(obj) is not dict:
    print('Only works for dictionaries'); return 
  if len(obj['files']) != len(np.unique(obj['files'])):
    print('Duplicates found in files!')
  unique_files = len(np.unique(obj['files']))
  for k in obj.keys():
    if k != 'files' and k != 'cellTypes' and k != 'cellType':
      if len(obj[k]) != len(np.unique(obj[k])):
        print('Duplicates found in %s!' %k)
      unique_items = len(np.unique(obj[k]))
  try:
    unique_cells = len(np.unique(obj['cellTypes']))
  except:
    unique_cells = len(np.unique(obj['cellType']))
  print('Contents: %i unique files, %i unique items, %i cell types'
         %(unique_files, unique_items, unique_cells))
  return


#######################################################################
# branch angles

def dist3(pt0, pt1):
  if len(pt0) == len(pt1) and len(pt0) == 3:
    return math.sqrt(sum([(pt0[i]-pt1[i])**2 for i in range(3)]))
  else:
    print('dimension mismatch')
    print(pt0, pt1)



def get_angle(pt0, midpt, pt1):
  if pt0 in [midpt, pt1] or pt1 in [midpt, pt0] or midpt in [pt0,pt1]:
    print('Some points are the same!')
    print(pt0, midpt, pt1)
  PT0 = dist3(pt1, midpt)
  PT1 = dist3(pt0, midpt)
  MIDPT = dist3(pt0, pt1)
  try:
    ang = math.acos( (MIDPT**2 - PT1**2 - PT0**2) / (2*PT1*PT0) )
    ang = ang*180/math.pi
  except:
    ang = 'nan'
  return ang



def find_points(seg0, seg1):
  seg0list, seg1list = [], []
  pt0where, pt1where, midwhere = None, None, None
  switchdict = {0: -1, -1: 0}
  # make a list of the node locations
  for n in seg0.nodes:
    seg0list.append([n.x,n.y,n.z])
  for n in seg1.nodes:
    seg1list.append([n.x,n.y,n.z])
    # find the common node, then use that to find the distinct ones
  for n in seg0list:
    if n in seg1list:
      midpt = n
  if seg0list.index(midpt) != 0:
    pt0where = 0
    pt0 = seg0list[0]
  else:
    pt0where = -1
    pt0 = seg0list[-1]
  if seg1list.index(midpt) != 0:
    pt1where = 0
    pt1 = seg1list[0]
  else:
    pt1where = -1
    pt1 = seg1list[-1]
  
  f = True
  if pt0 == pt1 or pt0==midpt:
    f = False
    if pt0where == 0:
      try:
        pt0=seg0list[1]
        f = True
      except:
        pass
    elif pt0where == -1:
      try:
        pt0=seglist[-2]
        f = True
      except:
        pass
  if pt0 == pt1 or pt1==midpt:
    if pt1where == 0:
      try:
        pt1=seg1list[1]
        f = True
      except:
        pass
    elif pt1where == -1:
      try:
        pt1=seg1list[-2]
        f = True
      except:
        pass
  if f == False:
    print('Tried to find new coordinates, but failed. Skipping')
  
  if pt0 in [midpt, pt1] or pt1 in [midpt, pt0] or midpt in [pt0,pt1]:
    print(seg0list, seg1list)
  #print('pt0 at %i, pt1 at %i' %(pt0where, pt1where))
  if pt1 and pt0 and midpt:
    return pt0, midpt, pt1
  else:
    print('could not figure out segments %s and %s' %(seg0.name, seg1.name))
    print(seg0list, seg1list)
    return [False]


def branch_angles(geo):
  angles = []
  for b in geo.branches:
    for n in b.neighbors:
      pts = find_points(n, b)
      if len(pts) == 3:
        pt0, midpt, pt1 = pts[0], pts[1], pts[2]
      angles.append(get_angle(pt0, midpt, pt1))
  angles = [a for a in angles if a!='nan']
  with open('temp_angles.txt', 'w') as fOut:
    for a in angles:
      fOut.write('%.10f, \n' %a)
  return angles



#######################################################################
# path length and tortuosity

def path_lengths(geo):
  tips, tipinds = geo.getTipIndices()
  pDF = PathDistanceFinder(geo, geo.soma, 0.5)
  tipsegs = [geo.segments[i] for i in tips]
  path = [pDF.distanceTo(x,y) for x, y in zip(tipsegs, tipinds)]
  tort = [pDF.tortuosityTo(x,y) for x, y in zip(tipsegs, tipinds)]
  return path, tort


def path_lengths2(geo):
  # if FilamentIndex != geo.segments[index], use this: 
  tips, tipinds = geo.getTipIndices()
  tipsegs = [i for i in geo.segments if geo.getFilamentIndex(i) in tips]
  pDF = PathDistanceFinder(geo, geo.soma, 0.5)
  path, tort = [], []
  for x, y in zip(tipsegs, tipinds):          
    try:
      p, t = pDF.distanceTo(x,y), pDF.tortuosityTo(x,y)
      path.append(p)
      tort.append(t)
    except:
      continue
  return path, tort



#######################################################################
# sholl stuff

def interpoint_dist(geo):
  # determine the distances between successive points
  def nodex(node):
    return [node.x, node.y, node.z]
  dists = []
  for s in geo.segments:
    for n in range(len(s.nodes)-1):
      dists.append(dist3(nodex(s.nodes[n]), nodex(s.nodes[n+1])))
  print('Mean distance (%i points): %.5f +/- %.5f' 
         %(len(dists), np.mean(dists), np.std(dists)))
  return dists


def interpolate_nodes(geo):
  # find the most common distance betwixt successive nodes and then,
  # when successive nodes leave integer multiples of this distance
  # interpolate the difference to 'even' it out
  def nodex(node):
    return [node.x, node.y, node.z]
    
  def interp(pt1, pt2, ints):
    Xs = np.linspace(pt1[0], pt2[0], ints)
    Ys = np.linspace(pt1[1], pt2[1], ints)
    Zs = np.linspace(pt1[2], pt2[2], ints)
    return [[Xs[i],Ys[i],Zs[i]] for i in range(len(Xs))]
    
  dist = np.median(interpoint_dist(geo))
  pts = []
  segcount = -1
  for s in geo.segments:
    segcount = segcount + 1
    if segcount % 100 == 0:
      print('Completed %i/%i segments ' 
             %(segcount,len(geo.segments)))
    for n in range(len(s.nodes)-1):
      # if too far between nodes, interpolate
      if dist3(nodex(s.nodes[n]),nodex(s.nodes[n+1])) > 2 * dist:
        integer_interpolate = int((dist3(nodex(s.nodes[n]),
                                         nodex(s.nodes[n+1])))
                                   /dist)
        new_pts = interp(nodex(s.nodes[n]),nodex(s.nodes[n+1]),
                         integer_interpolate)
      # else just add the regular node pts
      else:
        new_pts = [nodex(s.nodes[n]), nodex(s.nodes[n+1])]
      # add the points as long as they don't already exist in pts
      for p in new_pts:
        if p not in pts:
          pts.append(p)
  # now should have all the points
  soma = geo.soma.coordAt(0.5)
  distances = []
  for p in pts:
    distances.append(dist3(soma, p))
  return distances


######################################################################
# partition asymmetry

def get_segment(geo, segname):
  for s in geo.segments:
    if s.name == segname:
      return s


def tips_asymmetry(geo):
  # Get the tip asymmetry of the neuron. Follow the soma's neighbors
  # until there are more than 1, then start there.
  # seg_lengths: dict with a section_name for keys and float as values
  # seg_tips: dict with sec_name as key and list of segment objects as values
  prevsegs = [geo.soma]
  newsegs = [i for i in geo.soma.neighbors if i not in prevsegs]
  go = True
  while go:
    if len(newsegs) > 1:
      nebs = newsegs
      go = False
    else:
      for k in newsegs:
        prevsegs.append(k)
        for j in k.neighbors:
          newsegs.append(j)
        # not sure if this is allowed, but should be since not referencing by index
        newsegs.pop(newsegs.index(k))
        
  pDF = PathDistanceFinder(geo, geo.soma, 0.5)
  # nebs = geo.soma.neighbors
  tips, tipPositions = geo.getTips()
  seg_names = {}
  seg_tips = {}
  for n in nebs:
    seg_names[n.name] = []
    seg_tips[n.name] = []
  seg_lengths = {}
  for t, pos in zip(tips, tipPositions):
    curr_path = pDF.pathTo(t, pos)
    for n in seg_names.keys():
      # if the bifurcation in question is contained in the path soma->tip
      if get_segment(geo,n) in curr_path:
        # add this tip to n
        seg_tips[n].append(t)
        for c in curr_path:
          if c not in seg_names[n]:
            seg_names[n].append(c)
  # now should have all of the segments that lead to the tips in each key
  for k in seg_names.keys():
    seg_lengths[k] = 0
    for s in seg_names[k]:
      seg_lengths[k] = seg_lengths[k] + s.length
  
  return seg_lengths, seg_tips


def tip_coords(geo, seg_tips):
  # return x-y-z tuples for each tip; just use the (1) position of each tip seg
  tip_coords = {}
  for k in seg_tips.keys():
    tip_coords[k] = []
    for t in seg_tips[k]:
      tip_coords[k].append(t.coordAt(1))
  return tip_coords
  
  


######################################################################
# torques

def getNormVector(points):
  #print(points, np.shape(points))
  v1 = [points[1][0][i] - points[0][0][i] for i in range(3)]
  v2 = [points[2][0][i] - points[0][0][i] for i in range(3)]
  normVec = np.cross(v1,v2)
  return normVec


def angleBetween(plane1,plane2,planCoords):
  # get normal vectors
  n1, n2 = getNormVector(planCoords[plane1]), \
           getNormVector(planCoords[plane2])
  angle = np.arccos( (abs(n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2])) /
                     ( np.sqrt(n1[0]**2+n1[1]**2+n1[2]**2) *
                       np.sqrt(n2[0]**2+n2[1]**2+n2[2]**2) ) )
  return angle*180/np.pi


def get_torques(geo):
  # return bifurcation torques
  Cons =  geo.connections
  Seg1s, Seg2s = [], []
  for c in Cons:
    Seg1s.append(c['filament1']) # here, location1 is always 0
    Seg2s.append(c['filament2']) # here, location2 is always 1
    #geometry.c['filament1'].coordAt(c['location1'])
  
  tsegs = np.array([Seg1s,Seg2s]).T
  tsegs = tsegs.reshape(len(tsegs)*2)
  segs = set(tsegs)
  planCoords = {}
  
  count = 0
  for seg in segs:
    friends, friendcoords = [], []
    for s in geo.segments:
      if s.name == seg:
        friends.append(s.name)
        if s.name in Seg1s:
          friends.append(Seg2s[Seg1s.index(s.name)])
        if s.name in Seg2s:
          friends.append(Seg1s[Seg2s.index(s.name)])
    #print('friends compiled')
      
    for s in geo.segments:
      if s.name in friends:
        friendcoords.append([s.coordAt(1)])
    count = count + 1
    #if count%100 == 0:
    #  print('%i of %i segments done' %(count, len(segs)))
    if len(friendcoords) > 2: # need 3 points to define plane
      planCoords[seg]=friendcoords
  
  planCoordskeys = []
  for s in geo.segments: # loop through segments to find plane-neighbors
    if s.name in planCoords.keys():
      for n in s.neighbors:
        if n.name in planCoords.keys(): # if the neighbor is also a bifurcation
          planCoordskeys.append([s.name, n.name]) # add it
        else: # otherwise, keep looking for a neighbor that is
          for nn in n.neighbors:
            if nn.name in planCoords.keys():
              planCoordskeys.append([s.name, nn.name])
  
  # get torques
  torques = []
  for P in planCoordskeys:
    torques.append(angleBetween(P[0],P[1],planCoords))
  return torques



###############################################################################
# Ellipse fitting, distance to nearest point stuff


def getNoSomaPoints(geo):
  # get the downsampled nodes, but not the soma
  somaPos = geo.soma.coordAt\
            (geo.soma.centroidPosition(mandateTag='Soma'))
  print('Soma position: %.5f, %.5f, %.5f' %(somaPos[0],somaPos[1],somaPos[2])) # works
  nodes = []
  for seg in geo.segments:
    if 'Soma' not in seg.tags:
      nodes.append(seg.coordAt(0))
      nodes.append(seg.coordAt(0.5))
      nodes.append(seg.coordAt(1))
  print('Sampled %i nodes' %len(nodes))
  
  return nodes



def findBounds(nodelist):
  # return the x,y,z bounds of the node list
  xs, ys, zs = [], [], []
  
  for n in range(len(nodelist)):
    xs.append(nodelist[n][0])
    ys.append(nodelist[n][1])
    zs.append(nodelist[n][2])

  bounds = {'xmin': min(xs), 'xmax': max(xs), 
            'ymin': min(ys), 'ymax': max(ys),
            'zmin': min(zs), 'zmax': max(zs)}
  
  return bounds



def getGridPoints(nodelist, pplot=False):
  # create a grid around the neuropil and use linspace to fill the volume
  bounds = findBounds(nodelist)
  gridpoints = []
  xs = np.linspace(bounds['xmin'], bounds['xmax'], 10)
  ys = np.linspace(bounds['ymin'], bounds['ymax'], 10)
  zs = np.linspace(bounds['zmin'], bounds['zmax'], 10)
  spacing = xs[1]-xs[0]
  
  # 1000 grid volume points
  for i in range(len(xs)-1):
    for j in range(len(ys)-1):
      for k in range(len(zs)-1):
        gridpoints.append([(xs[i]+xs[i+1])/2,
                           (ys[j]+ys[j+1])/2,
                           (zs[k]+zs[k+1])/2])
  print('gridpoints is length %i' %len(gridpoints))
  
  boxx, boxy, boxz = [], [], []
  for g in range(len(gridpoints)):
    boxx.append(gridpoints[g][0])
    boxy.append(gridpoints[g][1])
    boxz.append(gridpoints[g][2])
  
  nodex, nodey, nodez = [], [], []
  for n in range(len(nodelist)):
    nodex.append(nodelist[n][0])
    nodey.append(nodelist[n][1])
    nodez.append(nodelist[n][2])
  
  if pplot:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
  #ax.plot(boxx, boxy)
    ax1.scatter(boxx, boxy, boxz, color='r', marker='.', alpha=0.5)
    ax1.scatter(nodex, nodey, nodez, color='k', marker='.', alpha=1)
  # ax.set_xlabel('')
  # plt.show()
    
  return gridpoints, spacing
  


def closestPoint(rectpoint, nodes):
  # find the closest neuron node to a rectangle point
  ptmin = np.inf
  ptind, pt = None, None
  for n in range(len(nodes)):
    dist = dist3(rectpoint, nodes[n])
    if dist < ptmin:
      ptmin = dist
      ptind = n
      pt = nodes[n]
  return ptind, ptmin


def closestPointPool(things):
  # find the closest neuron node to a rectangle point
  # things[0] = rect point, things[1] = all nodes
  things[0] = rectpoint
  things
  ptmin = np.inf
  ptind, pt = None, None
  for n in range(len(nodes)):
    dist = dist3(rectpoint, nodes[n])
    if dist < ptmin:
      ptmin = dist
      ptind = n
      pt = nodes[n]
  return ptmin # does not return pt_index


def getSurfacePoints(gridpoints, nodes, spacing, pplot=False):
  # given volume points and neuropil nodes, create downsampled
  # volume of the neuropil (if a neuron point is in a given cube, 
  # the cube is a 1, else 0
  ellipsePoints = []
  if type(gridpoints) is not np.ndarray:
    gridpoints = np.array(gridpoints)
  if type(nodes) is not np.ndarray:
    nodes = np.array(nodes)
  
  for b in range(len(gridpoints)):
    _, dist = closestPoint(gridpoints[b], nodes)
    if dist <= spacing/8.:
      ellipsePoints.append(gridpoints[b])
    if b % 100 == 0 and b != 0:
      print('%i/%i points examined' %(b, len(gridpoints)))
      
  print('Now have %i neuropil points' %len(ellipsePoints))
  
  surfx, surfy, surfz = [], [], []
  for s in ellipsePoints:
    surfx.append(s[0])
    surfy.append(s[1])
    surfz.append(s[2])
  if pplot:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(surfx, surfy, surfz, color='g', marker='.')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    plt.show()
  
  return ellipsePoints


def writeFile(points, outfile):
  # write points to a ascii; this is generally not necessary
  if outfile is None:
    outfile = 'neuropil_surfpoints.txt'  
  with open(outfile, 'w') as fOut:
    for p in range(len(points)):
      # print(points[p])
      ptstring = [str(points[p][0]), str(points[p][1]), str(points[p][2])]
      ptstr = ' '.join(ptstring)
      fOut.write(ptstr)
      fOut.write('\n')
      #print
  fOut.close()
  print('%s file written.' %outfile)
  return


# Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1

def give_ellipse(axes, shrink, translate):
  """
  axes: [1x3], shrink: scalar (ratio), translate: [1x3]
  Returns a 2-D ellipse of points when given the 3 axes ([maj, min, 3rd])
  and where on the 3rd axis the current slice is
  --> axes = original evals ### scale omitted here
  --> 'shrink' is the ratio that determines 
      how large how the ellipse should be stretched in 2-D
  --> axes[2] not used in this version
  """
  norm_ax = [i/max(axes) for i in axes]
  xs = np.linspace(-norm_ax[0],norm_ax[0],1000)
  ys = [np.sqrt( (1 - (i**2/norm_ax[0])) * norm_ax[1] ) for i in xs]
  # get rid of the nans
  opts = [[x,y] for x,y in zip(xs,ys) if np.isfinite(y)]
  # need to get the negative part of the y half of the graph
  pts = []
  for p in opts:
    pts.append([p[0],-p[1]])
    pts.append(p)
  # pts are currently the 'largest' possible, need to shrink by 'where'
  pts = np.array(pts)
  pts = pts * shrink
  newpts = []
  for p in pts:
    _pt = [axes[0] * p[0] + translate[0],  \
           axes[1] * p[1] + translate[1],  \
           translate[2]]
    if _pt not in newpts:
      newpts.append(_pt)
  
  return newpts


def get_reduced_points(geo, outfile=None):
  # only pre-req is to run getNoSomaPoints first
  nodes = getNoSomaPoints(geo)
  gridpoints, spacing = getGridPoints(nodes)
  ellipsePoints = getSurfacePoints(gridpoints, nodes, spacing)
  #writeFile(ellipsePoints, outfile)
  
  return ellipsePoints


def check_eigen(s_vals, s_vecs, pts):
  """
  For singular value decomposition, check the orientations of vectors
  vs. the points they're supposed to represent
  """
  # Get zero-centered points first
  #means = [pts[i] for i in range(len(pts)) if i%100==0] # downsample
  means = pts
  _m = [np.mean([j[0] for j in means]), np.mean([j[1] for j in means]),
        np.mean([j[2] for j in means])]
  # subtract the mean but keep the shape
  newmeans = []
  for m in means:
    newmeans.append([m[0]-_m[0],m[1]-_m[1],m[2]-_m[2]])
  dmax = farthest_pt(pts)
  # get eigenvectors normalized by distance from farthest pts
  scales = [i/max(s_vals)*dmax for i in s_vals]
  print(scales)
  v1 = [[0,0,0],[scales[0]*s_vecs[0][0], scales[0]*s_vecs[1][0], 
                 scales[0]*s_vecs[2][0]]]
  v2 = [[0,0,0],[scales[1]*s_vecs[0][1], scales[1]*s_vecs[1][1], 
                 scales[1]*s_vecs[2][1]]]
  v3 = [[0,0,0],[scales[2]*s_vecs[0][2], scales[2]*s_vecs[1][2], 
                 scales[2]*s_vecs[2][2]]]
  print(v1,v2,v3)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  for m in newmeans:
    ax.scatter(m[0],m[1],m[2], c='b', edgecolor='b', alpha=0.2)
  ax.plot([0,v1[1][0]], [0,v1[1][1]], [0,v1[1][2]], c='r')
  ax.plot([0,v2[1][0]], [0,v2[1][1]], [0,v2[1][2]], c='g')
  ax.plot([0,v3[1][0]], [0,v3[1][1]], [0,v3[1][2]], c='k')
  plt.show()
  return newmeans


def build_ellipse(geo):
  """
  Uses singular values from a uniformly resampled neuron grid to get
  major/minor axes to create an ellipsoid; scales and translates the
  ellipsoid back to neuron space.
  """
  gpts = get_reduced_points(geo)
  gmean = [np.mean([i[0] for i in gpts]),
           np.mean([i[1] for i in gpts]),
           np.mean([i[2] for i in gpts])]
  # get singular values
  _, s_vals, s_vecs = np.linalg.svd(gpts)
  s = np.array([i/max(s_vals) for i in s_vals])
  # scale singular values by longest distance
  dmax = farthest_pt(gpts)
  s = s * dmax
  # hyperbolic scaling reference for taper of top/bottom
  _x = np.linspace(0,10,50)
  _y = -_x**2 + 100
  y = [i/max(_y) for i in _y]
  y.reverse()
  zscale = [i for i in y]
  y.reverse()
  for i in y:
    zscale.append(i)
  eig_pts = []
  # make 100 layers of v3
  zlayers = np.linspace(-s[2],s[2],100)
  for v in zlayers:
    newpts = give_ellipse(s, zscale[list(zlayers).index(v)], 
                          [0,0,0])
    for p in newpts:
      eig_pts.append(p)
  eig_pts = np.array(eig_pts)
  # now have all eigen points, need to re-orient axes
  pts = eig_pts.dot(np.linalg.inv(s_vecs))
  # now translate:
  pts = [[p[0]+gmean[0], p[1]+gmean[1], p[2]+gmean[2]] for p in pts]
  return pts, gpts, eig_pts
  
    
def get_distances(geo, multi=None):
  """
  Return the "distances", the distance from each ellipse point to the
  closest point of the neuron's skeleton.
  """
  if multi is None:
    ellipse_pts, _, _ = build_ellipse(geo)
    nodes = getNoSomaPoints(geo)
    distances = []
    ellipse_pts = ellipse_pts[::100]
    for e in ellipse_pts:
      _, d = closestPoint(e, nodes)
      if ellipse_pts.index(e)%100==0:
        print('%i (of %i) points sampled' %(ellipse_pts.index(e), len(ellipse_pts)))
      distances.append(d)
    return distances
  elif type(multi) is int:
    from multiprocessing import Pool
    p = Pool(multi)
    # distances = pool.map(closestPointPool, 
  return distances
  

#######################################################################
# simple branch stuff

def branch_lengths(geo, locations=False):
  lengths = [b.length for b in geo.branches]
  locations = [b.coordAt(0.5) for b in geo.branches]
  if locations:
    return lengths, locations
  else:
    return lengths


def branch_order(geo):
  return [b.branchOrder for b in geo.branches]
    



  





































