import numpy as np
import cv2
import os
import math
from detection import *
import csv
import matplotlib.pyplot as plt
import urllib.request
from numba import jit
import sys

def face_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #haar_classifier = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    #face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)
    face = segment(image)

    if face != -1:
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]

        arr = [x,y,w,h]
        dim = (100, 100)
        img = image_gray[y:y + w, x:x + h]
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized, arr
    else:
        return -1

@jit
def face_detection2(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_classifier = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

    face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)

    if len(face) != 0:
        (x, y, w, h) = face[0]

        dim = (100, 100)
        img = image_gray[y:y + w, x:x + h]
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized, face[0]
    else:
        return -1

def face_detection3(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #haar_classifier = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    #face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)
    face = imageEnhance(image)

    if face != -1:
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]

        arr = [x,y,w,h]
        dim = (100, 100)
        img = image_gray[y:y + w, x:x + h]
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized, arr
    else:
        return -1


# Apply LBP Algorithm on a block of the image
def get_lbp_hist(grayscale_img):
    dim = grayscale_img.shape

    lbp = np.zeros(256)

    for i in range(1, dim[0] - 1):
        for j in range(1, dim[1] - 1):
            pixel = grayscale_img[i][j]
            neighbours = np.concatenate(
                [grayscale_img[i - 1][j - 1:j + 2], [grayscale_img[i][j + 1]], grayscale_img[i + 1][j - 1:j + 2][::-1],
                 [grayscale_img[i][j - 1]]])
            neighbours[neighbours <= pixel] = '0'
            neighbours[neighbours > pixel] = '1'
            binary = ''.join(str(c) for c in neighbours)

            binary = int(binary, 2)
            lbp[binary] += 1

    return lbp  # <LBP histogram which is a list of 256 numbers>


# Segment the image into 7x7 blocks,apply LBP algorithm on them, then Concatente all hitograms into one
def segment_img(img):
    dim = img.shape
    patch_width = 10
    patch_height = 10

    histograms = np.zeros(20736)
    count = 0
    for x in range(0, dim[0], patch_width):
        for y in range(0, dim[1], patch_height):

            if (patch_width + x >= dim[0] or patch_width + y >= dim[1]):
                continue
            patch = img[x:x + patch_width, y:y + patch_height].copy()

            histograms[count:count+256] = get_lbp_hist(patch)   
            count+=256

    #histo = np.concatenate(histograms)

    return histograms


def train_data():
    filename = "training.csv"
    file = open(filename, "w")
    csvwriter = csv.writer(file)

    for person in os.listdir('training'):

        # classes.append(person)

        for image in os.listdir('training/' + person):
            img_path = 'training/' + person + '/' + image
            img = cv2.imread(img_path)
            dim = img.shape
            img = cv2.resize(img, ( int(dim[1]*0.5),int(dim[0]*0.5)))
            result = face_detection2(img)
            if result == -1:
                continue
            else:
                img = result[0]
            histo = segment_img(img)
            row = [person]
            for x in histo:
                row.append(x)

            csvwriter.writerow(row)
    file.close()


def read_data():
    classes = []
    train_hist = []
    train_labels = []

    file = open("training.csv", "r")
    lines = file.readlines()
    for l in lines:
        line = l.split(',')

        name = line[0]
        hist = []

        toint = int(float(line[2]))
        if name not in classes:
            classes.append(name)

        for i in range(1, len(line)):
            toint = int(float(line[i]))
            hist.append(toint)
        train_hist.append(hist)
        train_labels.append(name)

    file.close()
    train_hist = np.array(train_hist)
    train_labels = np.array(train_labels)
    classes = np.array(classes)

    return classes, train_hist, train_labels


# Apply Nearest Neighbour Algorithm for test image
def classify(img, face=-1):
    if face == -1:
        img, _ = face_detection(img)

    test_hist = segment_img(img)
    #print(test_hist)
    mini_dist = 1000000
    mini_class = -1

    distances = np.sum(np.abs(train_hist[:] - test_hist), axis=1)
    # distances = np.sum( (train_hist[:] - test_hist)**2, axis=1)
    distances = np.sqrt( distances )

    index = np.argmin(distances)
    x = (   distances[index]  /256)

    mini_class = train_labels[index]
    
    if x > 0.32 :
        print("Error : "+mini_class + " : " +str(x))
        return -1

    print("Correct : "+mini_class + " : " +str(x))
    return mini_class


def test_img(img, face=-1):
    c = classify(img, face)
    if c != -1:
        return c
    else:
        return "No Match"

def live_stream():
    url="http://192.168.1.2:8080/shot.jpg"

    while True:

        # Use urllib to get the image from the IP camera
        imgResponse = urllib.request.urlopen(url)

        # Numpy to convert into a array
        imgNp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)

        # Decode the array to OpenCV usable format
        img = cv2.imdecode(imgNp,-1)
        dim = img.shape
        img = cv2.resize(img, (dim[1]*0.5,dim[0]*0.5))

        # put the image on screen

        #############################################
        # img = rotate(img,270)
        ##############################################
        try:
            res = face_detection2(img)
        except:
            break
        # res = -1
        if res != -1:
            img2 = res[0]
            face = res[1]
            (x, y, w, h) = face

            c = test_img(img2, 1)
            # print(w*h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img = cv2.putText(img, c, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 2)

        else:
            print("no")
        cv2.imshow('Face Recognition', img)
        # cv2.imshow("a",img)

        # Program closes if q is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break

def rotate(img,angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def get_video(name):

    cap = cv2.VideoCapture("Test/"+name)

    while (cap.isOpened()):
        ret, img = cap.read()
        dim = img.shape
        img = cv2.resize(img, ( int(dim[1]*0.5),int(dim[0]*0.5)))
            
        #############################################
        # img = rotate(img,270)
        ##############################################
        try:
            res = face_detection2(img)
        except:
            break
        # res = -1
        if res != -1:
            img2 = res[0]
            face = res[1]
            (x, y, w, h) = face

            c = test_img(img2, 1)
            # print(w*h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img = cv2.putText(img, c, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 2)

        else:
            print("no")
        cv2.imshow('Face Recognition', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    cap.release()
    cv2.destroyAllWindows()

# train_data()

print(type(int(sys.argv[1])))

classes, train_hist, train_labels =read_data()

t = int(sys.argv[1])
if t == 1:
    live_stream()
else:
    name = sys.argv[2]
    get_video(name)


# img = cv2.imread("1.jpg")
# img = face_detection2(img)
# if img != -1:
#     img = img[0]    
#     # cv2.imwrite("res.jpg",img)
#     name = test_img(img, 1)
#     print(name)
# else:
#     print("sorry :(")