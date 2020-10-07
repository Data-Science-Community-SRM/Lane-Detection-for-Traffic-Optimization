import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import warnings

def grayscale(new_image):

	img_array = np.array(new_image.convert('RGB'))

	image =  cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

	return image

def gauss(new_image):

    image = grayscale(new_image)

    return cv2.GaussianBlur(image, (5, 5), 0)

def canny(new_image , min = 50, max = 120):
    
    first_canny = gauss(new_image)
    canny = cv2.Canny(first_canny ,min , max)
    
    return canny

def region_of_interest_masked_for_video(new_image):

	image =  cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(image , (5,5), 0)
	canny = cv2.Canny(blur ,40 , 120)
	height = canny.shape[0]
	polygons = np.array([[(200, height),(1100 , height), (550 , 250)]])  #Creating a triangular mask

	mask = np.zeros_like(canny)  #Black pixels

	cv2.fillPoly(mask , polygons , 255) #Fill mask with triangle dimensions as white(255)

	mask = cv2.bitwise_and(canny , mask)

	return mask

def region_of_interest_masked(new_image):
	
	image = canny(new_image , 50 , 120)
	height = image.shape[0]
	polygons = np.array([[(200, height),(1100 , height), (550 , 250)]])  #Creating a triangular mask

	mask = np.zeros_like(image)  #Black pixels

	cv2.fillPoly(mask , polygons , 255) #Fill mask with triangle dimensions as white(255)

	mask = cv2.bitwise_and(image , mask)

	return mask

def display_lines(image, lines):
    
    line_image = np.zeros_like(image)
    
    if lines is not None:
        
        for line in lines:
            
            x1 , y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1) ,(x2,y2), (255,0,0), 10)
               
    return line_image

def make_cordinates(image , line_parameters):
    
    try:
    	slope , intercept = line_parameters    
    except TypeError:
    	slope , intercept = 0.01 , 0.0

    y1 = image.shape[0]
    
    y2  = int(y1*(0.49))
    
    x1 = int((y1 - intercept)/slope)
    
    x2 = int((y2 - intercept)/slope)
    
    return np.array([x1, y1 , x2 , y2])


def average_slope_intercept(image , lines):
    
    left_fit = []
    right_fit = []
    
    for line in lines:
        
        x1, y1 , x2 , y2 = line.reshape(4)
        
        parameters = np.polyfit((x1 , x2), (y1, y2), 1)
        
        slope = parameters[0]
        
        intercept = parameters[1]
        
        if slope < 0:
            
            left_fit.append((slope , intercept))
        else:
            
            right_fit.append((slope , intercept))
            
    left_fit_average = np.average(left_fit , axis = 0)
    right_fit_average = np.average(right_fit , axis = 0)
        
    left_line = make_cordinates(image , left_fit_average)
        
    right_line = make_cordinates(image , right_fit_average)
    
    return np.array([left_line , right_line])





st.title("Lane Detection using OpenCV")
st.text("Built with streamlit")

activities = ["Image","Video"]

choice = st.sidebar.selectbox("Choose static or dynamic", activities)

st.set_option('deprecation.showfileUploaderEncoding', False)


if choice == "Image":

    st.subheader("Lane Detection on Image ")

    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

    our_image = Image.open(image_file)

    st.text('Original Image')

    options = ['Original','Grayscaling','Gaussian Blur' ,'CannyEdge','Masked Image','Final Image']

    types = st.selectbox('Choose how the feature to be applied', options)

    if(types == 'Original'):

        st.write("This is your original image")
        st.image(our_image, width = 800, caption = 'Fig.')

        st.write('Done!')

    elif(types == 'Grayscaling'):

        st.write("Hold Up, we are working on it!")

        gray_scaled_image =  grayscale(our_image)

        st.image(gray_scaled_image, width = 800, caption = 'Fig.')

        st.write('Done!')


    elif(types == 'Gaussian Blur'):

        st.write("Processing. . . Here is Your image")

        gaussian_blurred_image = gauss(our_image)
        st.image(gaussian_blurred_image, width = 800, caption = 'Fig.')
        st.success('Done!')


    elif(types == 'CannyEdge'):

        st.write('Pass the Parameters -')
        st.text('Preferably between 40 & 200')
        min_reso = st.number_input('Enter minimum :')
        max_reso = st.number_input('Enter maximum :')
        st.write("Here is the picture with the changes made")


        canny_edge_image = canny(our_image, min_reso, max_reso)
        st.image(canny_edge_image, width = 800, caption = 'Fig.')
        st.success('Done!')

    elif(types == 'Masked Image'):

        st.write('Processing request for Masked Image')
        masked_image = region_of_interest_masked(our_image)

        st.image(masked_image, width = 800, caption = 'Fig.')
        st.success('Done!')

    elif(types == 'Final Image'):

        st.write('Here comes the Final Image')

        lane_image = np.copy(our_image)
        masked_image = region_of_interest_masked(our_image)

        lines = cv2.HoughLinesP(masked_image , 2 , (np.pi)/180 , 100 , np.array([]), minLineLength = 40 , maxLineGap = 5)

        averaged_lines = average_slope_intercept(lane_image , lines)

        line_image = display_lines(our_image , averaged_lines)
        combo_image = cv2.addWeighted(lane_image , 0.8, line_image , 1, 1)
        
        st.image(combo_image , width = 800 , caption = 'Fig.')
        st.success('Done!')




    


elif choice == 'Video':

    st.subheader("Lane Detect on Video")

    video_file = st.file_uploader("Choose a video...", type="mp4")

    if video_file is not None:

        st.video(video_file)

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    vf = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while vf.isOpened():

        ret, frame = vf.read()

        # if frame is read correctly ret is True
        if not ret:
            break

     masked_image = region_of_interest_masked_for_video(frame)

     lines = cv2.HoughLinesP(masked_image , 2 , np.pi/180 , 100 , np.array([]), minLineLength = 40 , maxLineGap = 5)
     averaged_lines = average_slope_intercept(frame , lines)

     line_image = display_lines(frame, averaged_lines)

     combo_image = cv2.addWeighted(frame , 0.8, line_image , 1, 1)

     stframe.image(combo_image, width = 800)

vf.release()

st.write('Done!')




   