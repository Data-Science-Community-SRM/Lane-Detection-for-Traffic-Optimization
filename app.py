import streamlit as st
import cv2
from PIL import Image 
import numpy as np
import os
import tempfile

@st.cache
def grayscale(new_img):
    img_array = np.array(new_img.convert('RGB'))
    gray =  cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return gray

@st.cache
def gaussian_blur(new_img):
    grayscaled = grayscale(new_img)
    blur = cv2.GaussianBlur(grayscaled , (5,5), 0)
    return blur
	
	

@st.cache
def canny(new_img , min = 50, max = 120):
    canny1 = gaussian_blur(new_img)
    canny = cv2.Canny(canny1 ,min , max)
    
    
    return canny

@st.cache
def region_of_int_video(new_img):
    image =  cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image , (5,5), 0)
    canny = cv2.Canny(blur ,40 , 120)
    height = canny.shape[0]
    polygons = np.array([[(200, height),(1100 , height), (550 , 250)]])  #Creating a triangular mask
    mask = np.zeros_like(canny)  #Black pixels
    cv2.fillPoly(mask , polygons , 255) #Fill mask with triangle dimensions as white(255)
    mask = cv2.bitwise_and(canny , mask)
    return mask

@st.cache
def region_of_int(new_img):
    image = canny(new_img , 50 , 120)
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

st.title("Lane-line Detection App")
st.text("Build with Streamlit and OpenCV")

activities = ['Image' , 'Video']
choice = st.selectbox('Choose the input type ', activities)

if choice == 'Image':
    image_file = st.file_uploader("Choose an image", type=["jpeg","jpg","png"])
    if image_file is not None:
        our_image = Image.open(image_file)
        
    
    result_type = st.sidebar.radio("Result Type",["Original", "Grayscale", "Gaussian Blur", "Canny edge", "Masked", "Final"])  
    if result_type == 'Original':
        st.write('Here is your original image - ')
        st.image(our_image, width = 500, height = 300)
        st.success('Done!')
    

    elif result_type == 'Grayscale':
        st.write('Here is your grayscaled image - ')
        gray =  grayscale(our_image)
        st.image(gray, width = 500, height = 300)
        st.success('Done!')
    
    elif result_type == 'Gaussian Blur':
        st.write('Here is your Gaussian Blurred image - ')
        blur = gaussian_blur(our_image)
        st.image(blur, width = 500, height = 300)
        st.success('Done!')
    

    elif result_type == 'Canny edge':
        canny_image = canny(our_image)
        st.image(canny_image, width = 500, height = 300)
        st.success('Done!')

    elif result_type == 'Masked':
        st.write('Here is your masked image -')
        masked_image = region_of_int(our_image)
        st.image(masked_image , width = 500, height = 300)
        st.success('Done!')
		

    elif result_type == 'Final':
        st.write('Here is your final image detecting lane-lines!')
        lane_image = np.copy(our_image) 
        masked_image = region_of_int(our_image)
        lines = cv2.HoughLinesP(masked_image , 2 , (np.pi)/180 , 100 , np.array([]), minLineLength = 40 , maxLineGap = 5)
        averaged_lines = average_slope_intercept(lane_image , lines)
        line_image = display_lines(our_image , averaged_lines)
        combo_image = cv2.addWeighted(lane_image , 0.8, line_image , 1, 1)
        st.image(combo_image , width = 500, height = 300)
        st.success('Done!')
        st.balloons()

elif choice == 'Select Video':
    video_file = st.file_uploader("Choose a video...", type="mp4")
    if video_file is not None:
        st.video(video_file)

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())


    vf = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
	    
        if not ret:
            break
	    
        masked_image = region_of_int_video(frame)
        lines = cv2.HoughLinesP(masked_image , 2 , np.pi/180 , 100 , np.array([]), minLineLength = 40 , maxLineGap = 5)
        averaged_lines = average_slope_intercept(frame , lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame , 0.8, line_image , 1, 1)
        stframe.image(combo_image, width = 800)

    vf.release()

    st.write('Done!')
    st.balloons()
	
	
	