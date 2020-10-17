# Lane Detection
In this project, we'll be detecting lane lines in videos and images of roads using the Computer Vision. Following lane lines is one of the most important traffic rules, so detecting them is a significant task while building models for autonomous(self-driving) vehicles. This project was hosted using Streamlit & deployed using Heroku.

## Requirements
```
opencv-python==4.2.0.44 or above
numpy==1.19.2 or above
matplotlib==3.1.1
Pillow==7.2.0
```
## Concepts

### Image Used
![alt text](https://github.com/Data-Science-Community-SRM/Lane-detection-for-traffic-optimization/blob/master/Images/test_image.jpg)

1. **Grayscaling**

Grayscale is a range of shades of gray without apparent color. The darkest possible shade is black, which is the total absence of transmitted or reflected light. The lightest possible shade is white, the total transmission or reflection of light at all visible wavelength s. Intermediate shades of gray are represented by equal brightness levels of the three primary colors (red, green and blue) for transmitted light, or equal amounts of the three primary pigments (cyan, magenta and yellow) for reflected light.

![alt text](https://github.com/Data-Science-Community-SRM/Lane-detection-for-traffic-optimization/blob/master/Images/Grayscaling.jpeg)


2. **Gaussian Blur**

In image processing, a Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function. It is a widely used effect in graphics software, typically to reduce image noise and reduce detail. The visual effect of this blurring technique is a smooth blur resembling that of viewing the image through a translucent screen, distinctly different from the bokeh effect produced by an out-of-focus lens or the shadow of an object under usual illumination.

![alt text](https://github.com/Data-Science-Community-SRM/Lane-detection-for-traffic-optimization/blob/master/Images/GaussianBlur.jpeg)

3. **Canny Edge Detection**

The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. The Canny filter is a multi-stage edge detector. It smooths the image with a Gaussian filter to reduce noise and unwanted details and textures.


![alt text](https://github.com/Data-Science-Community-SRM/Lane-detection-for-traffic-optimization/blob/master/Images/CannyEdgeDetection.jpeg)

After removing the noise -

![alt text](https://github.com/Data-Science-Community-SRM/Lane-detection-for-traffic-optimization/blob/master/Images/FilteredCannyImage.jpeg)

4. **Hough Transforms**

The Hough transform is a technique which can be used to isolate features of a particular shape within an image. Because it requires that the desired features be specified in some parametric form, the classical Hough transform is most commonly used for the detection of regular curves such as lines, circles, ellipses, etc. 
A generalized Hough transform can be employed in applications where a simple analytic description of a feature(s) is not possible. The main advantage of the Hough transform technique is that it is tolerant of gaps in feature boundary descriptions and is relatively unaffected by image noise.

A good reference to understand Hough Line Transforms -> https://www.youtube.com/watch?v=4zHbI-fFIlI



## Final Processed Image

![alt text](https://github.com/Data-Science-Community-SRM/Lane-detection-for-traffic-optimization/blob/master/Images/FinalProcessedImage.jpeg)

