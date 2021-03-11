import os
import numpy as np
import sys
import math
import time
import zlib
import scipy.io

FETCH_BATCH_SIZE=32
BATCH_SIZE=32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
OUTPUTPOINTS=1024
REEBSIZE=1024

# thread_1 = list(range(8,50)) #mdl26 --> mdl29
# thread_1 = list(range(59,100)) #mdl30 --> mdl28
# thread_1 = list(range(109,150)) #mdl31 --> mdl28
thread_1 = list(range(159,200)) #mdl32 --> mdl29
# thread_1 = list(range(200,250)) #mdl28
# thread_1 = list(range(250,300)) #mdl29

folderLst = list(map(str,thread_1))
datadir = "../../data/"
datasavedir = "./mat_dir/1/"

def save_file(array, name):
    save_path = os.path.join(datasavedir,'%s.mat'%(name))
    scipy.io.savemat(save_path, {name[:-3]: array})
    print 'Saved %s.mat'%(name)

def main():
    
    bno = 1
    
    path = os.path.join(datadir,'%d/%d.gz'%(bno//1000,bno))
    
    # Reading the zipped file
    binfile=zlib.decompress(open(path,'r').read())

    # Saving the binary values into corresponding arrays + PreProcessing
    p=0
    color=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH,3))
    save_file(color,'raw_color_01')
    p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*3
    
    depth=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*2],dtype='uint16')
    save_file(depth,'depth_before_reshape_02')
    depth = depth.reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH))
    save_file(depth,'depth_after_reshape_03')
    p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*2
    
    rotmat=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*3*3*4],dtype='float32')
    save_file(rotmat,'rotmat_before_reshape_04')
    rotmat = rotmat.reshape((FETCH_BATCH_SIZE,3,3))
    save_file(rotmat,'rotmat_after_reshape_05')
    p+=FETCH_BATCH_SIZE*3*3*4
    
    ptcloud=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*POINTCLOUDSIZE*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,POINTCLOUDSIZE,3))
    save_file(ptcloud,'ptcloud_before_normalize_06')
    # ptcloud = ptcloud/2 # downsampling pointcloud ground truth --> FFK
    ptcloud=ptcloud.astype('float32')/255
    
    
    ## Point Cloud Processing
    beta=math.pi/180*20
    viewmat=np.array([[
        np.cos(beta),0,-np.sin(beta)],[
        0,1,0],[
        np.sin(beta),0,np.cos(beta)]],dtype='float32')
    save_file(viewmat,'viewmat_07')
    
    rotmat=rotmat.dot(np.linalg.inv(viewmat))
    save_file(rotmat,'rotmat_after_dot_08')
    
    for i in xrange(FETCH_BATCH_SIZE):
        ptcloud[i]=((ptcloud[i]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
    save_file(ptcloud,'ptcloud_after_transformation_09')
    p+=FETCH_BATCH_SIZE*POINTCLOUDSIZE*3
    
    
    reeb=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*REEBSIZE*2*4],dtype='uint16')
    save_file(reeb,'reeb_before_reshape_10')
    reeb = reeb.reshape((FETCH_BATCH_SIZE,REEBSIZE,4))
    save_file(reeb,'reeb_after_reshape_11')
    p+=FETCH_BATCH_SIZE*REEBSIZE*2*4
    
    keynames=binfile[p:].split('\n')
    save_file(keynames,'keynames_12')
    
    reeb=reeb.astype('float32')/65535
    for i in xrange(FETCH_BATCH_SIZE):
        reeb[i,:,:3]=((reeb[i,:,:3]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
    save_file(reeb,'reeb_after_transformation_13')
    
    data=np.zeros((FETCH_BATCH_SIZE,HEIGHT,WIDTH,4),dtype='float32')
    save_file(data,'image_empty_14')
    data[:,:,:,:3]=color*(1/255.0)
    save_file(data,'image_RGBonly_15')
    data[:,:,:,3]=depth==0
    save_file(data,'image_RGBD_16')
    validating=np.array([i[0]=='f' for i in keynames],dtype='float32')
    save_file(validating,'validating_keys_17')
    # deliverables are data, ptcloud, validating
        
        
if __name__ == '__main__':
    main()            
            

            