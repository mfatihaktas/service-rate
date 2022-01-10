import numpy as np
import scipy
from scipy import spatial
from scipy.spatial import HalfspaceIntersection
import itertools, string

import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from debug_utils import *
from plot_utils import *
import plot_polygon

class BucketConfInspector_wCode(object):
  def __init__(self, m, C, G, obj_bucket_m):
    self.m = m # number of buckets
    self.C = C # capacity of each bucket
    self.G = G
    self.obj_bucket_m = obj_bucket_m
    
    ## Columns of G are (k x 1) 0-1 vectors and represent the stored objects.
    ## Object-i is stored in Bucket-obj_bucket_m[i].
    self.k = G.shape[0]
    self.n = G.shape[1]
    
    ## repgroup's are given in terms of obj ids, in the sense of columns of G or keys of obj_bucket_m
    self.sys__repgroup_l_l = self.get_sys__repgroup_l_l()
    log(DEBUG, "", sys__repgroup_l_l=self.sys__repgroup_l_l)
    
    print('sys-repgroup_l, in terms of node ids:')
    for sys, repgroup_l in enumerate(self.sys__repgroup_l_l):
      str_ = "["
      for rg in repgroup_l:
        rg_str = '({})'.format(','.join(['{}'.format(obj_bucket_m[obj_id] ) for obj_id in rg] ) )
        str_ += rg_str + ', '
      str_ += ']'
      print('sys= {}, {}'.format(sys, str_) )
    
    # '''
    ## T
    _rg_l = []
    for srg_l in self.sys__repgroup_l_l:
      _rg_l.extend(srg_l)
    
    self.l = len(_rg_l)
    self.T = np.zeros((self.k, self.l))
    i = 0
    for s, rg_l in enumerate(self.sys__repgroup_l_l):
      j = i + len(rg_l)
      self.T[s, i:j] = 1
      i = j
    ## M
    self.M = np.zeros((m, self.l))
    for obj in range(self.n):
      for rgi, rg in enumerate(_rg_l):
        if obj in rg:
          self.M[obj_bucket_m[obj], rgi] = 1
    ## Halfspaces
    halfspaces = np.zeros((self.m+self.l, self.l+1))
    for r in range(self.m):
      halfspaces[r, -1] = -self.C
    halfspaces[:self.m, :-1] = self.M
    for r in range(self.m, self.m+self.l):
      halfspaces[r, r-self.m] = -1
    # log(INFO, "halfspaces= \n{}".format(halfspaces) )
    
    feasible_point = np.array([self.C/self.l]*self.l)
    self.hs = HalfspaceIntersection(halfspaces, feasible_point)
    # '''
  
  def __repr__(self):
    return 'BucketConfInspector_wCode[m= {}, C= {}, \nG=\n {}, \nobj_bucket_m= {}, \nM=\n{}, \nT=\n{}]'.format(self.m, self.C, self.G, self.obj_bucket_m, self.M, self.T)
  
  def to_sysrepr(self):
    sym_l = string.ascii_lowercase[:self.k]
    node_l = [[] for _ in range(self.m) ]
    for obj in range(self.n):
      ni = self.obj_bucket_m[obj]
      l = []
      for r in range(self.k):
        if self.G[r, obj] != 0:
          num = int(self.G[r, obj] )
          l.append('{}{}'.format(num, sym_l[r] ) if num != 1 else '{}'.format(sym_l[r] ) )
      node_l[ni].append('+'.join(l) )
    return str(node_l)
  
  def get_sys__repgroup_l_l(self):
    def does_right_contain_left(t1, t2):
      for e in t1:
        if e not in t2:
          return False
      return True
    
    sys__repgroup_l_l = [[] for s in range(self.k) ]
    for s in range(self.k):
      # y = self.G[:, s].reshape((self.k, 1))
      y = np.array([0]*s + [1] + [0]*(self.k-s-1) ).reshape((self.k, 1))
      repgroup_l = []
      for repair_size in [1, 2, 3]: # range(1, self.k+1): # [1, 2]:
        for subset in itertools.combinations(range(self.n), repair_size):
          ## Check if subset contains any previously registered smaller repair group
          skip = False
          for rg in repgroup_l:
            if does_right_contain_left(rg, subset):
              skip = True
              break
          if skip:
            continue
          l = [self.G[:, i] for i in subset]
          # A = np.array(l).reshape((self.k, len(l) ))
          A = np.column_stack(l)
          # print("\n")
          x, residuals, _, _ = np.linalg.lstsq(A, y)
          residuals = y - np.dot(A, x)
          # log(INFO, "", A=A, y=y, x=x, residuals=residuals)
          if np.sum(np.absolute(residuals) ) < 0.0001: # residuals.size > 0 and 
            repgroup_l.append(subset)
      sys__repgroup_l_l[s] = repgroup_l
    # blog(sys__repgroup_l_l=sys__repgroup_l_l)
    return sys__repgroup_l_l

  def plot_cap(self, d):
    if self.k == 2:
      self.plot_cap_2d(d)
    elif self.k == 3:
      self.plot_cap_3d(d)
    else:
      log(ERROR, "not implemented for k= {}".format(self.k) )
  
  def plot_cap_2d(self, d):
    # print("hs.intersections= \n{}".format(self.hs.intersections) )
    x_l, y_l = [], []
    for x in self.hs.intersections:
      y = np.matmul(self.T, x)
      x_l.append(y[0] )
      y_l.append(y[1] )
    
    points = np.column_stack((x_l, y_l))
    hull = scipy.spatial.ConvexHull(points)
    for simplex in hull.simplices:
      simplex = np.append(simplex, simplex[0] ) # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
      plot.plot(points[simplex, 0], points[simplex, 1], c=NICE_BLUE, marker='o')
    
    plot.legend()
    fontsize = 18
    plot.xlabel(r'$\rho_a$', fontsize=fontsize)
    plot.xlim(xmin=0)
    plot.ylabel(r'$\rho_b$', fontsize=fontsize)
    plot.ylim(ymin=0)
    plot.title(r'$k= {}$, $m= {}$, $C= {}$, $d= {}$'.format(self.k, self.m, self.C, d) + ', Volume= {0:.2f}'.format(hull.volume) \
      + '\n{}'.format(self.to_sysrepr() ), fontsize=fontsize, y=1.05)
    plot.savefig('plot_cap_2d.png', bbox_inches='tight')
    plot.gcf().clear()
    log(INFO, "done.")
  
  def plot_cap_2d_when_k_g_2(self):
    fontsize = 18
    
    xi_yi_l = list(itertools.combinations(range(self.k), 2) )
    fig, axs = plot.subplots(len(xi_yi_l), 1, sharex='col')
    figsize = [4, len(xi_yi_l)*4]
    for i, (xi, yi) in enumerate(xi_yi_l):
      print(">> i= {}, (xi, yi)= ({}, {})".format(i, xi, yi) )
      x_l, y_l = [0], [0]
      for _p in self.hs.intersections:
        p = np.matmul(self.T, _p)
        include = True
        for j in range(self.k):
          if j == xi or j == yi:
            continue
          if p[j] > 0.01:
            include = False
            break
        if include:
          x_l.append(p[xi] )
          y_l.append(p[yi] )
    
      ax = axs[i]
      plot.sca(ax)
      
      # print("x_l= {}".format(x_l) )
      # print("y_l= {}".format(y_l) )
      # print("Right before plotting red cross; i= {}".format(i) )
      # plot.plot([1], [1], c='red', marker='x', ms=12)
      # print("Right after plotting red cross; i= {}".format(i) )
      # plot.plot(x_l, y_l, c=NICE_BLUE, marker='o', ls='None')
      
      points = np.column_stack((x_l, y_l))
      # print("points= {}".format(points) )
      hull = scipy.spatial.ConvexHull(points)
      for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0] ) # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
        plot.plot(points[simplex, 0], points[simplex, 1], c=NICE_BLUE, marker='o')
      plot.xlim(xmin=0)
      plot.ylim(ymin=0)
      plot.xlabel('{}'.format(chr(ord('a') + xi) ), fontsize=fontsize)
      plot.ylabel('{}'.format(chr(ord('a') + yi) ), fontsize=fontsize)
      plot.title('i= {}'.format(i) )
    plot.suptitle(r'$k= {}$, $n= {}$, $C= {}$'.format(self.k, self.m, self.C) \
      + '\n{}'.format(self.to_sysrepr() ), fontsize=fontsize) # , y=1.05
    fig.set_size_inches(figsize[0], figsize[1] )
    plot.savefig('plot_cap_2d_when_k_g_2.png', bbox_inches='tight')
    plot.gcf().clear()
    log(INFO, "done.")
  
  def plot_cap_3d(self, d):
    ax = plot.axes(projection='3d')
    
    x_l, y_l, z_l = [], [], []
    for x in self.hs.intersections:
      y = np.matmul(self.T, x)
      x_l.append(y[0] )
      y_l.append(y[1] )
      z_l.append(y[2] )
    
    points = np.column_stack((x_l, y_l, z_l))
    hull = scipy.spatial.ConvexHull(points)
    for simplex in hull.simplices:
      simplex = np.append(simplex, simplex[0] ) # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
      ax.plot3D(points[simplex, 0], points[simplex, 1], points[simplex, 2], c=NICE_BLUE, marker='o')
    
    faces = hull.simplices
    verts = points
    triangles = []
    for s in faces:
      sq = [
        (verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]),
        (verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]),
        (verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]) ]
      triangles.append(sq)
    
    new_faces = plot_polygon.simplify(triangles)
    for sq in new_faces:
      f = mpl_toolkits.mplot3d.art3d.Poly3DCollection([sq] )
      # f.set_color(matplotlib.colors.rgb2hex(scipy.rand(20) ) )
      f.set_color(next(dark_color_c) )
      f.set_edgecolor('k')
      f.set_alpha(0.15) # 0.2
      ax.add_collection3d(f)
    
    plot.legend()
    fontsize = 18
    ax.set_xlabel(r'$\rho_a$', fontsize=fontsize)
    ax.set_xlim(xmin=0)
    ax.set_ylabel(r'$\rho_b$', fontsize=fontsize)
    ax.set_ylim(ymin=0)
    ax.set_zlabel(r'$\rho_c$', fontsize=fontsize)
    ax.set_zlim(zmin=0)
    ax.view_init(20, 30)
    # plot.title(r'$k= {}$, $m= {}$, $C= {}$, $d= {}$'.format(self.k, self.m, self.C, d) + ', Volume= {0:.2f}'.format(hull.volume) \
    #   + '\n{}'.format(self.to_sysrepr() ), fontsize=fontsize, y=1.05)
    plot.title(r'$k= {}$, $n= {}$, $d= {}$'.format(self.k, self.m, d), fontsize=fontsize)
    plot.gcf().set_size_inches(7, 5)
    plot.savefig('plot_cap_3d_{}.png'.format(self.to_sysrepr() ), bbox_inches='tight')
    plot.gcf().clear()
    log(INFO, "done.")





  
