import show3d
import show3d_balls
import numpy as np
import sys
a=np.loadtxt(sys.argv[1])
# show3d.showpoints(a)
show3d_balls.showpoints(a,showrot=True,ballradius=5)