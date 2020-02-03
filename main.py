#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:15:14 2020

@author: aidancookson
"""

import cv2
import time
import mss
import numpy as np
from matplotlib import pyplot as plt

HOOP_OFFSET_Y = 50

template_ball = cv2.imread('template_ball.png')
template_hoop = cv2.imread('template_hoop.png')

monitor = {"top": 60, "left": 430, "width": 500, "height": 890}

cv2.namedWindow('Screen', cv2.WINDOW_NORMAL)


test_img = cv2.imread('example.png')#[monitor['left']:monitor['left']+monitor['width'], monitor['top']:monitor['top']+monitor['height']]
                       #monitor['left']:monitor['left']+monitor['width']]

#cv2.resizeWindow('Screen', (monitor['width']//2, monitor['height']//2))
cv2.resizeWindow('Screen', (test_img.shape[0] // 2, test_img.shape[1]//2))


def get_hoop_loc(img):
    w, h = template_hoop.shape[:2]
    res = cv2.matchTemplate(img, template_hoop, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    center = (top_left[0] + w//2  - HOOP_OFFSET_Y, top_left[1] + h//2)
    return center

    
def get_ball_loc(img):
    w, h = template_ball.shape[:2]
    res = cv2.matchTemplate(img, template_ball, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    center = (top_left[0] + w//2, top_left[1] + h//2)
    return center

center = get_ball_loc(test_img)
center2 = get_hoop_loc(test_img)

img = test_img.copy()
cv2.circle(img,center, 8, 255, 2)
cv2.circle(img,center2, 8, (0, 255, 0), 2)

cv2.imshow('Screen', img)
cv2.waitKey(0)
    

'''
with mss.mss() as sct:
    # Part of the screen to capture

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))

        # Display the picture
        cv2.imshow("Screen", img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
            '''