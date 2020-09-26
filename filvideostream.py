#!/usr/bin/env python
# coding: utf-8

# In[2]:


from threading import Thread
import sys
import cv2


# In[4]:


if sys.version_info>=(3,0):
    from queue import Queue
else:
    from Queue import Queue


# In[7]:


class FileVideoStream:
    def __init__(self,path,queueSize=128):
        self.stream=cv2.VideoCapture(path)#instantiates our cv2.videocapture object
        self.stopped=False#if threading process should be stopped
        self.Q=Queue(maxsize=queueSize)
    
    def start(self):
        #start thread to read frames from the file video stream
        t=Thread(target=self.update,args=())
        t.daemon=True
        t.start()
        return self
    def update(self):
        #keep looping infinitely
        while True:
            #if the thread indicator variable is set,stop it
            if self.stopped:
                return
            #otherwise ensure the queue has room in it
            if not self.Q.full():
                #read the next frame from the file
                (grabbed,frame)=self.stream.read()
                #if grabbed boolean is false,then
                #we have reached the end 
                
                if not grabbed:
                    self.stop()
                    return
                #add the frame to the queue
                self.Q.put(frame)
    def read(self):
        #return the next frame in the queue
        return self.Q.get()
    def more(self):
        #return True if there are still frames in queue
        return self.Q.qsize()>0
    def stop(self):
        #indicate that the thread should be stopped
        self.stopped=True
                
                
    


# In[ ]:




