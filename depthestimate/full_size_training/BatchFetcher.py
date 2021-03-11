import sys
import numpy as np
import cv2
import random
import math
import os
import time
import zlib
import socket
import threading
import Queue
import sys
import cPickle as pickle
# import show3d

FETCH_BATCH_SIZE=32
BATCH_SIZE=32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
OUTPUTPOINTS=1024
REEBSIZE=1024

class BatchFetcher(threading.Thread):
	def __init__(self, dataname):
		super(BatchFetcher,self).__init__()
		self.queue=Queue.Queue(64)
		self.stopped=False
		self.datadir = dataname
		self.bno=0
	def work(self,bno):
		path = os.path.join(self.datadir,'%d/%d.npz'%(bno//1000,bno))
		if not os.path.exists(path):
			self.stopped=True
			print "error! data file not exists: %s"%path
			print "please KILL THIS PROGRAM otherwise it will bear undefined behaviors"
			assert False,"data file not exists: %s"%path

		npzfile = np.load(path)
	        data = npzfile['data']
        	ptcloud = npzfile['ptcloud']
       		validating = npzfile['validating']
        
		return (data,ptcloud,validating)
	def run(self):
		while self.bno<300000 and not self.stopped:
			self.queue.put(self.work(self.bno%300000))
			self.bno+=1
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	def shutdown(self):
		self.stopped=True
		while not self.queue.empty():
			self.queue.get()

if __name__=='__main__':
	dataname = "YTTRBtraindump_220k"
	fetchworker = BatchFetcher(dataname)
	fetchworker.bno=0
	fetchworker.start()
	for cnt in xrange(100):
		data,ptcloud,validating = fetchworker.fetch()
		validating = validating[0]!=0
		assert len(data)==FETCH_BATCH_SIZE
		# for i in range(len(data)):
			# cv2.imshow('data',data[i])
			# while True:
				# cmd=show3d.showpoints(ptcloud[i])
				# if cmd==ord(' '):
					# break
				# elif cmd==ord('q'):
					# break
			# if cmd==ord('q'):
				# break


