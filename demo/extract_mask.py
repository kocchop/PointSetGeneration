import numpy as np
import cv2
import sys

path = sys.argv[1]
depth = cv2.imread(path)
name_prefix = path[:-10]

mask = np.where(depth>0,0,255)

cv2.imwrite("%s_m.png"%name_prefix, mask) 