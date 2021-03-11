import show3d
import show3d_balls
import get_frames
import numpy as np
import sys
import cv2

GRAYSCALE = 0
path = sys.argv[1]
a=np.loadtxt(path)
name_prefix = path[:-8]
# show3d.showpoints(a)
# show3d_balls.showpoints(a,showrot=True,ballradius=5)
showsz, frames = get_frames.showpoints(a,showrot=True,ballradius=8)
# print len(frames)
out = cv2.VideoWriter('%s_video.avi'%name_prefix,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (showsz, showsz))
# out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'X264'), 10, (showsz, showsz))
 
for i in range(len(frames)):
    # cv2.imwrite("./save/%d.png"%i, frames[i])
    img_r = frames[i][:,:,0]
    img_g = frames[i][:,:,1]
    img_b = frames[i][:,:,2]
    
    mask_r = np.where(img_r>0,1,0)
    back_r = np.where(img_r==0,GRAYSCALE,0)
    
    mask_g = np.where(img_g>0,1,0)
    back_g = np.where(img_g==0,GRAYSCALE,0)
    
    mask_b = np.where(img_b>0,1,0)
    back_b = np.where(img_b==0,GRAYSCALE,0)
   
    cshow = cv2.applyColorMap(frames[i], cv2.COLORMAP_RAINBOW)
    cshow[:,:,0] = cshow[:,:,0]*mask_r + back_r 
    cshow[:,:,1] = cshow[:,:,1]*mask_g + back_g 
    cshow[:,:,2] = cshow[:,:,2]*mask_b + back_b 
    
    # print "img {}, mask {}, back {}, cshow {}".format(img.shape, mask.shape, background.shape, cshow.shape)
    # exit()
    out.write(cshow)
    
out.release()
