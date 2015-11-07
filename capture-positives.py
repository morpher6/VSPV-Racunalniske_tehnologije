"""Raspberry Pi Face Recognition Treasure Box
Positive Image Capture Script
Copyright 2013 Tony DiCola 

Run this script to capture positive images for training the face recognizer.
"""
import glob
import os
import sys
import select

import cv2

import config
import face
import time
from stick import SenseStick

# Prefix for positive training image filenames.
POSITIVE_FILE_PREFIX = 'positive_'

dir_name = raw_input("Please enter user_name (dir_name): ")
#dir_name = 'bostjan'

positive_dir = config.POSITIVE_DIR +'/'+ dir_name

if __name__ == '__main__':
    camera = config.get_camera()
    # Create the directory for positive training images if it doesn't exist.
    if not os.path.exists(positive_dir):
        os.makedirs(positive_dir)
    # Find the largest ID of existing positive images.
    # Start new images after this ID value.
    files = sorted(glob.glob(os.path.join(positive_dir, 
        POSITIVE_FILE_PREFIX + '[0-9][0-9][0-9].pgm')))
    count = 0
    if len(files) > 0:
        # Grab the count from the last filename.
        count = int(files[-1][-7:-4])+1
    print 'Capturing positive training images.'
    print 'Press button or type c (and press enter) to capture an image.'
    print 'Press Ctrl-C to quit.'
    camera.preview()
    while True:
        # Check if button was pressed or 'c' was received, then capture image.
        if config.is_letter_input('c'):
            print 'Capturing image...'
            image = camera.read()
            # Convert image to grayscale.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Get coordinates of single face in captured image.
            result = face.detect_single(image)
            if result is None:
                print 'Could not detect single face!  Check the image in capture.pgm' \
                      ' to see what was captured and try again with only one face visible.'
                continue
            x, y, w, h = result
            # Crop image as close as possible to desired face aspect ratio.
            # Might be smaller if face is near edge of image.
            crop = face.crop(image, x, y, w, h)
            # Save image to file.
            filename = os.path.join(positive_dir, POSITIVE_FILE_PREFIX + '%03d.pgm' % count)
            cv2.imwrite(filename, crop)
            print 'Found face and wrote training image', filename
            count += 1
