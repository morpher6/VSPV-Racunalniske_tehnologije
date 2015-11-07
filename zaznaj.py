"""Raspberry Pi Face Recognition Treasure Box
Treasure Box Script
Copyright 2013 Tony DiCola 
"""
import cv2

import config
import face
import os
import time

from sense_hat import SenseHat

sense = SenseHat()

X = [255, 0, 0]  # Red
G = [0, 255, 0]  # Red
O = [0, 0, 0]  #Black
W = [255, 255, 255]  # White

tick = [
O, O, O, O, O, O, O, O,
O, O, O, O, O, O, O, G,
O, O, O, O, O, O, G, O,
O, O, O, O, O, G, O, O,
O, O, G, O, G, O, O, O,
O, O, O, G, O, O, O, O,
O, O, O, O, O, O, O, O,
O, O, O, O, O, O, O, O
]

cross = [
O, O, O, O, O, O, O, O,
O, X, O, O, O, O, X, O,
O, O, X, O, O, X, O, O,
O, O, O, X, X, O, O, O,
O, O, O, X, X, O, O, O,
O, O, X, O, O, X, O, O,
O, X, O, O, O, O, X, O,
O, O, O, O, O, O, O, O
]

if __name__ == '__main__':
    # Initialize camer.
    camera = config.get_camera()
    # Move box to locked position.
    print 'Press button to lock (if unlocked), or unlock if the correct face is detected.'
    print 'Press Ctrl-C to quit.'

    # Load training data into model
    models = []
    for root, dirs, files in os.walk(config.POSITIVE_DIR):
        for user in dirs:
            model = cv2.createEigenFaceRecognizer()
            # Test face against model.
            print 'Loading training data for', user
            model.load(user+'__'+config.TRAINING_FILE)
            print 'Training data loaded!'
            models.append([user, model])

    # Check for the positive face and unlock if found.
    print "capture face to test"
    camera.preview()
    while True:

        # Check if capture should be made.
        # TODO: Check if button is pressed.
        if config.is_letter_input('c'):
            print 'Button pressed, looking for face...'
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
            # Crop and resize image to face.
            crop = face.resize(face.crop(image, x, y, w, h))

            found = False

            for user, model in models:
                label, confidence = model.predict(crop)
                print 'Predicted {0} face with confidence {1} (lower is more confident).'.format(
                    'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', 
                    confidence), 'for user', user
                if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
                    print 'Recognized face! Hello', user
                    found = True
                    sense.set_pixels(tick)
                    time.sleep(3)
                    sense.show_message("Hi "+user+"!")
                    sense.clear()
                    break

            if not found:
                print 'Did not recognize face'
                sense.set_pixels(cross)
                time.sleep(3)
                sense.clear()

            print "Capture another face to test"
