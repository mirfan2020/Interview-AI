from PIL import ImageDraw,Image
import face_recognition
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fer import FER
img = Image.open('user_photo.png')
''''img = img.resize((800, 600), Image.ANTIALIAS)
img = img.convert('RGB')
#img.show()
data = np.array(img)
locations = face_recognition.face_locations(data)
print(locations)'''
#img_copy.show()
pil_image = Image.open("user_photo.png").convert('RGB') 
open_cv_image = np.array(pil_image) 
# Convert RGB to BGR 
img_copy_fer = open_cv_image[:, :, ::-1].copy() 
detector = FER()
locations = detector.detect_emotions(img_copy_fer)
print(locations)
locations =  [data['box'] for data in locations]
# We copy the image since we don't want to draw on the original image.
img_copy = img_copy_fer.copy()
draw = ImageDraw.Draw(img_copy)
for location in locations:
    top, right, bottom, left = location
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=3)
del(draw)

img_copy.show()
