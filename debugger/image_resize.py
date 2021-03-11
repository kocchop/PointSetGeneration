import os
import numpy as np
import sys
import math
import time
import zlib
import cv2
from PIL import Image
import scipy.ndimage as ndi

FETCH_BATCH_SIZE=32
BATCH_SIZE=32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
OUTPUTPOINTS=1024
REEBSIZE=1024

datadir = "../../data/"

def img_show():
    

def main():
    
    bno = 0
    
    path = os.path.join(datadir,'%d/%d.gz'%(bno//1000,bno))
    
    # Reading the zipped file
    binfile=zlib.decompress(open(path,'r').read())

    # Saving the binary values into corresponding arrays + PreProcessing
    p=0
    color=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH,3))
    
    
    print "The input shape is {}".format(color.shape)
    
    # try:
        # cv_resize = cv2.resize(color, dsize=(color.shape[1]/2, color.shape[2]/2), interpolation=cv2.INTER_LINEAR)
        # print "After downsampling with opencv, the shape became {}".format(cv_resize.shape)
    # except:
        # print "opencv didn't work!"
    
    # try:
        # img = Image.fromarray(color[1,:,:,:])
        # print "After converting with Pillow, the shape became {}".format(img.shape)
        # pil_resize = img.resize((96,128))
        # print "After downsampling with Pillow, the shape became {}".format(pil_resize.shape)
    # except:
        # print "Pillow didn't work!"
    
    try:
        imgs = ndi.zoom(color, (1, 0.5, 0.5, 1), order=2)
        print "After converting with SciPy Zoom, the shape became {}".format(imgs.shape)
        
        
        # pil_resize = img.resize((96,128))
        # print "After downsampling with Pillow, the shape became {}".format(pil_resize.shape)
    except:
        print "Scipy Zoom didn't work!"
    
        
        
if __name__ == '__main__':
    main()            
            

            

