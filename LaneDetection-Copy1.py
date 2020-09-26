#!/usr/bin/env python
# coding: utf-8

# In[3]:


from filvideostream import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


def final(img):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss=cv2.GaussianBlur(grey,(7,7),0.5)
    canny=cv2.Canny(gauss,50,150)
    return canny


# In[5]:


def region_of_interest(image):
    height=image.shape[0]
    width=image.shape[1]
    triangle=[(100,height),
                        (475,325),(width,height)]
    mask=np.zeros_like(image)#return an array of zeros( black(0) mask)
    mask=cv2.fillPoly(mask,np.array([triangle],np.int32),255)
    mask=cv2.bitwise_and(image,mask)
    return mask



# In[6]:


def make_coordinates(isolated,average):
    slope,y_int=average
    y1=isolated.shape[0]
    y2=int((y1*3/5))
    x1=int((y1-y_int)//slope)#floor division(return int)
    x2=int((y2-y_int)//slope)
    return np.array([x1,y1,x2,y2])

def average(image,lines):
    left=[]
    right=[]
    for line in lines:
        print(line)
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        y_intercept=parameters[1]
        if slope<0:
            left.append((slope,y_intercept))
        else:
            right.append((slope,y_intercept))
    left_fit_average=np.average(left,axis=0)
    right_fit_average=np.average(right,axis=0)
    left_line=make_coordinates(isolated,left_fit_average)
    right_line=make_coordinates(isolated,right_fit_average)
    return np.array([left_line,right_line])



# In[7]:


def display_lines(image,lines):
    lines_image=np.zeros_like(image)#create a balcked out image
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line
            cv2.line(lines_image,(x1,y1),(x2,y2),(0,0,255),10)#red line
    return lines_image


# In[8]:


ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=False,
                help="path to image file")
ap.add_argument("-v","--video",required=False,
                help="path to input video file")
args=vars(ap.parse_args())
if args["image"]:

    #for image
    img=cv2.imread(args["image"])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#changing color space

    copy = np.copy(img)
    image1= final(img)
    isolated = region_of_interest(image1)
    #cv2.imshow("image1", image1)
    #cv2.imshow("iso", isolated)
    #cv2.waitKey(0)
    #DRAWING LINES: (order of params) --> region of interest,
    #bin size (P, theta), min intersections needed, placeholder array,

    lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_line = average(copy, lines)
    black_lines = display_lines(copy, average_line)

    #taking wighted sum of original image and lane lines image

    lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
    cv2.imshow("lanes", lanes)
    cv2.waitKey(0)
elif args["video"]:

    #for video
    #start the file video stream thread and allow the buffer to
    #start to fill
    print("[INFO] starting video file thread")
    fvs=FileVideoStream(args["video"]).start()
    time.sleep(1.0)
    #start the fps timer
    fps=FPS().start()

    #loop over frames from the video file stream


    while fvs.more():
        #grab the frame from the threaded video file stream,resize
        #it and convert it to greayscale(while still retaining three channels)
        frame=fvs.read()
        image=final(frame)
        isolated=region_of_interest(image)
        lines = cv2.HoughLinesP(isolated, 4, np.pi/180, 60, np.array([]), minLineLength=1, maxLineGap=5)
        average_line=average(frame,lines)
        black_lines=display_lines(frame,average_line)
        lanes=cv2.addWeighted(frame,0.8,black_lines,1,1)
        frame=np.dstack([frame,frame,frame])

        #display the size of queue on the frame
        #show the fram and update the fps counter
        cv2.imshow("Frame",lanes)
        cv2.waitKey(1)
        fps.update()
        #stop the timer and display fps info
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()


# In[9]:





# In[11]:





# In[ ]:





# In[ ]:
