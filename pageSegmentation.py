# import the necessary packages
import argparse
import os
import time

# https://stackoverflow.com/questions/47260277/cluster-bounding-boxes-and-draw-line-on-them-opencv-python
import cv2
import numpy as np
import tesserocr as tr
from PIL import Image
from scanFixed import scan
from scanFixed import scanFromFourPoint


from word import word


# #################################### PAGE SEGMENTATION ################################# 
# def pageSegmentation(image):
#     cv_img = cv2.imread(image)
    
#     cv_img = scan(cv_img)

#     # filename1 = './images/croped/warpedabc.jpg' 
#     # cv2.imwrite(filename1, cv_img)

#     # since tesserocr accepts PIL images, converting opencv image to pil
#     pil_img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))

#     #initialize api
#     api = tr.PyTessBaseAPI()
#     try:
#         # set pil image for ocr
#         api.SetImage(pil_img)
#         # Google tesseract-ocr has a page segmentation methos(psm) option for specifying ocr types
#         # psm values can be: block of text, single text line, single word, single character etc.
#         # api.GetComponentImages method exposes this functionality
#         # function returns:
#         # image (:class:`PIL.Image`): Image object.
#         # bounding box (dict): dict with x, y, w, h keys.
#         # block id (int): textline block id (if blockids is ``True``). ``None`` otherwise.
#         # paragraph id (int): textline paragraph id within its block (if paraids is True).
#         # ``None`` otherwise.
#         boxes = api.GetComponentImages(tr.RIL.TEXTLINE,True)
#         # get text
#         text = api.GetUTF8Text()
#         # iterate over returned list, draw rectangles
#         for (im,box,_,_) in boxes:
#             x,y,w,h = box['x'],box['y'],box['w'],box['h']
#             # cv2.rectangle(cv_img, (x-25,y-25), (x+w+25,y+h+25), color=(0,0,255))
            
#             imgSegmentation = cv_img[y-25:y+h+35, x:x+w]
#             # print("imgSegmentation.shape[:2] : " + str(imgSegmentation.shape[:2]))
            
#             filename = './images/croped/savedImage-' + str((x+y)) + '.jpg'
#             cv2.imwrite(filename, imgSegmentation)

#             # cv2.imshow("imgSegmentation", imgSegmentation)
#             # cv2.waitKey(0)

#             return imgSegmentation
#     finally:
#         api.End()

# #################################### SCAN ################################# 
def imgScan(image):
    imgSegmentation = cv2.imread(image)
    imgScan = scan(imgSegmentation)
    return imgScan


# #################################### MAIN RECOGNITION ################################# 
def pageSegmentation():

    image = "images/some_image.jpg"
    east =  "frozen_east_text_detection.pb"

    imgSegmentation = imgScan(image)
    if (imgSegmentation != 'cut'):
        # text_detection(imgSegmentation, east, min_confidence = 0.5, width=320, height=320, )  
        word(imgSegmentation) 
        
        # cv2.waitKey(0)
        return imgSegmentation
    elif(imgSegmentation == 'cut'):
        return imgSegmentation

def pageSegmentationFourPoint( x1, x2, x3, x4, y1, y2, y3, y4):
    image = "images/some_image.jpg"

    imgSegmentation = cv2.imread(image)
    # print(">>>>" + str(imgSegmentation.shape[:2]))
    imgSegmentation = scanFromFourPoint(imgSegmentation, x1, x2, x3, x4, y1, y2, y3, y4)

    word(imgSegmentation) 
    return imgSegmentation

