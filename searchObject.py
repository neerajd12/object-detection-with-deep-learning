import json
import numpy as np
import cv2
import pandas

from utils import imageUtils
from keras.optimizers import Adam
from keras.models import model_from_json
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip

image_utils = imageUtils()

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
        Draw boxes on the detected objects.
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def slide_window(img, x_start_stop=[None, None],
                      y_start_stop=[None, None],
                      xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
        Function to get a list of windows to search on image.
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def add_heat(heatmap, bbox_list):
    """
       Iterate through list of boxes and mark the detected boxes.
    """
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Box form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    """
        Threshhold the detections to reduce the number of detections.
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def search_windows(img, windows):
    """
        use the model to predict the objects and return the windows.
    """
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        image_array = image_utils.pre_process_image(img[window[0][1]:window[1][1], window[0][0]:window[1][0]])
        transformed_image_array = image_array[None, :, :, :]
        pred = model.predict(transformed_image_array, batch_size=1)
        max = np.argmax(pred)
        if max == 0 and pred[0][max] > 0.90:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def draw_labeled_bboxes(img, labels):
    """
        Draw the boxes around detected object.
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

def process_img(img):
    """
        Wrapper function.
    """
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    xy_win=int((img.shape[0]*0.9 - img.shape[0]*0.6)/2)
    windows = slide_window(img, x_start_stop=[0, img.shape[1]],
                                y_start_stop=[int(img.shape[0]*0.6), int(img.shape[0]*0.9)], 
                                xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(img, windows)
    heat = add_heat(heat,hot_windows)
    heat = apply_threshold(heat,2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    return draw_labeled_bboxes(np.copy(img), labels)
    

img = cv2.imread('test4.jpg')
draw_img=porcess_img(img)
#image_utils.plot_images([draw_img])

project_video_res = 'project_video_res.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_video_clip = clip1.fl_image(porcess_img)
project_video_clip.write_videofile(project_video_res, audio=False)

project_video_res = 'test_video_res.mp4'
clip1 = VideoFileClip("test_video.mp4")
project_video_clip = clip1.fl_image(porcess_img)
project_video_clip.write_videofile(project_video_res, audio=False)