import os
import numpy as np
import sys
import math
import time
import zlib

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
datasavedir = "../../data_np/"

def main():
    
    for folder in folderLst:
        print "Now in folder {}. Starting...".format(folder)
        
        # create the save directory
        save_dir = datasavedir + folder + '/'
        os.makedirs(save_dir)
        
        start_folder_time = time.time()
        
        data_dir = os.path.join(datadir,'%s/'%(folder))
        
        f = []
        
        for _,_,filename in os.walk(data_dir):
            f.extend(filename)
        
        folder_count = 0
        for file in f:        
            path = os.path.join(data_dir,file)
            
            # Reading the zipped file
            binfile=zlib.decompress(open(path,'r').read())

            # Saving the binary values into corresponding arrays + PreProcessing
            p=0
            color=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH,3))
            p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*3
            depth=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*HEIGHT*WIDTH*2],dtype='uint16').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH))
            p+=FETCH_BATCH_SIZE*HEIGHT*WIDTH*2
            rotmat=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*3*3*4],dtype='float32').reshape((FETCH_BATCH_SIZE,3,3))
            p+=FETCH_BATCH_SIZE*3*3*4
            ptcloud=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*POINTCLOUDSIZE*3],dtype='uint8').reshape((FETCH_BATCH_SIZE,POINTCLOUDSIZE,3))
            # ptcloud = ptcloud/2 # downsampling pointcloud ground truth --> FFK
            ptcloud=ptcloud.astype('float32')/255
            beta=math.pi/180*20
            viewmat=np.array([[
                np.cos(beta),0,-np.sin(beta)],[
                0,1,0],[
                np.sin(beta),0,np.cos(beta)]],dtype='float32')
            rotmat=rotmat.dot(np.linalg.inv(viewmat))
            for i in xrange(FETCH_BATCH_SIZE):
                ptcloud[i]=((ptcloud[i]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
            p+=FETCH_BATCH_SIZE*POINTCLOUDSIZE*3
            reeb=np.fromstring(binfile[p:p+FETCH_BATCH_SIZE*REEBSIZE*2*4],dtype='uint16').reshape((FETCH_BATCH_SIZE,REEBSIZE,4))
            p+=FETCH_BATCH_SIZE*REEBSIZE*2*4
            keynames=binfile[p:].split('\n')
            reeb=reeb.astype('float32')/65535
            for i in xrange(FETCH_BATCH_SIZE):
                reeb[i,:,:3]=((reeb[i,:,:3]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
            data=np.zeros((FETCH_BATCH_SIZE,HEIGHT,WIDTH,4),dtype='float32')
            data[:,:,:,:3]=color*(1/255.0)
            data[:,:,:,3]=depth==0
            validating=np.array([i[0]=='f' for i in keynames],dtype='float32')
            
            # deliverables are data, ptcloud, validating
            
            # save file name
            save_file_name = file[:-3] + '.npz'
            save_path = os.path.join(save_dir,save_file_name)
            
            # write to file
            np.savez(save_path, data=data, ptcloud=ptcloud, validating=validating)
        
        end_folder_time = time.time()
        folder_time = end_folder_time - start_folder_time
        folder_count = folder_count + 1
        
        print "The folder {} took {} mins for preprocessing. ok.".format(folder, folder_time/60.0)
    
    total_time = time.time() - global_start_time
    print "Total preprocessing time for {} folders is {} mins! Adios!!".format(folder_count,total_time/60.0)
        
if __name__ == '__main__':
    global_start_time = time.time()
    main()            
            

            
