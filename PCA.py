from tkinter import Toplevel, Label
import matplotlib.pyplot as plt  # plot import
import matplotlib.colors  # color import
import numpy as np  # importing numpy
from PIL import Image  # importing PIL to read all kind of images
from PIL import ImageTk
import glob
import cv2
import os
import csv
from detection import *


def face_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_classifier = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

    face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)
    if len(face) != 0:
        (x, y, w, h) = face[0]

        dim = (100, 100)
        # img = image_gray[y:y + w, x:x + h]
        img = image[y:y + w, x:x + h]
        resized = img
        return resized, face[0]
    else:
        return -1

def face_detection2(image):
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face = segment(image)
    if face != -1:
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]

        arr = [x,y,w,h]

        #dim = (100, 100)
        #img = image_gray[y:y + w, x:x + h]
        img = image[y:y + w, x:x + h]
        #resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img,arr
    else:
        return -1


def displaying_faces_grid(displaying_faces):
    size = 100, 100
    fig1, axes_array = plt.subplots(2, 2)
    fig1.set_size_inches(5, 5)
    count = 0
    for x in range(2):
        for y in range(2):
            print(count)
            print(displaying_faces[count])

            draw_image = displaying_faces[count]
            draw_image.thumbnail(size)
            draw_image = np.asarray(draw_image, dtype=float) / 255.0

            image_plot = axes_array[x][y].imshow(draw_image, cmap=plt.cm.gray)
            axes_array[x][y].axis('off')
            count = count + 1
    fig1.canvas.set_window_title('Displaying all faces')
    plt.show()


def display_mean_face(face_array):
    print(face_array.shape)
    mean = np.mean(face_array, 0)
    fig2, axes_array = plt.subplots(1, 1)
    fig2.set_size_inches(5, 5)
    image_plot = axes_array.imshow(mean, cmap=plt.cm.gray)
    fig2.canvas.set_window_title('mean faces')
    plt.show()
    return mean


def performing_pca(face_array):
    print("MEAN FACE DISPLAY")
    mean = display_mean_face(face_array)
    # flattening array
    flatten_Array = []
    for x in range(len(face_array)):
        flat_Array = face_array[x].flatten()
        flatten_Array.append(flat_Array)
    flatten_Array = np.asarray(flatten_Array)
    mean = mean.flatten()
    return mean, flatten_Array,


def display_all(images):
    fig3, axes_array = plt.subplots(2, 2)
    fig3.set_size_inches(5, 5)
    count = 0
    for x in range(2):
        for y in range(2):
            draw_image = images[count]
            image_plot = axes_array[x][y].imshow(draw_image, cmap=plt.cm.gray)
            axes_array[x][y].axis('off')
            count = count + 1
    fig3.canvas.set_window_title('Eigen Faces')
    plt.show()


def reading_faces_and_displaying():
    face_array = []
    train_labels = []
    displaying_faces = []
    dim = (100, 100)

    # Convert it to jpg
    count = 0
    for person in os.listdir("Train"):
        for image in os.listdir('Train/' + person):
            img_path = 'Train/' + person + '/' + image
            img = cv2.imread(img_path)
            
            result = face_detection(img)
            if result == -1:
                os.remove(img_path)
                continue
            #img = result[0]
            os.remove(img_path)
            cv2.imwrite("Train/"+person+"/im" + str(count) + ".jpg", img)
            count+=1
    #==========================

    for person in os.listdir("Train"):
        for face_images in glob.glob('Train/' + person + '/*.jpg'):
            face_image = Image.open(face_images)
            res = face_detection(np.asarray(face_image))
            if res == -1:
                print(person)
                continue
            face_image = Image.fromarray(res[0])
            face_image = face_image.resize((425, 425)).convert('L')
            displaying_faces.append(face_image)
            face_image = np.asarray(face_image, dtype=float) / 255.0
            face_array.append(face_image)
            train_labels.append(person)

    print(train_labels)
    # print(face_array)
    print("DISPLAYING ORIGINAL FACES")
    # displaying_faces_grid(displaying_faces)
    face_array = np.asarray(face_array)
    train_labels = np.asarray(train_labels)
    print(face_array.shape)

    return face_array, train_labels

def display_reconstruction(images):
    fig4, axes_array = plt.subplots(2, 2)
    fig4.set_size_inches(5, 5)
    count = 0
    for x in range(2):
        for y in range(2):
            draw_image = np.reshape(images[count, :], (425, 425))
            image_plot = axes_array[x][y].imshow(draw_image, cmap=plt.cm.gray)
            axes_array[x][y].axis('off')
            count = count + 1
    # fig4.canvas.set_window_title('Reconstructed faces for k=' + str(k))
    # plt.show()


def reconstructing_faces(k, mean, substract_mean_from_original, V):
    weights = np.dot(substract_mean_from_original, V.T)
    reconstruction = mean + np.dot(weights[:, 0:k], V[0:k, :])
    display_reconstruction(reconstruction)

def write_file(name,V):
    dim = V.shape
    file = open(name, "w")
    csvwriter = csv.writer(file)
    for i in range(dim[0]):
        row = []
        for j in range(dim[1]):
            row.append(V[i][j])
        csvwriter.writerow(row)
    file.close()
# k=2
# print("RECONSTRUCTING FACES FOR K=2")
# reconstructing_faces(k,mean,substract_mean_from_original,V)
# k=5
# print("RECONSTRUCTING FACES FOR K=5")
# reconstructing_faces(k,mean,substract_mean_from_original,V)
# k=15
# print("RECONSTRUCTING FACES FOR K=15")
# reconstructing_faces(k,mean,substract_mean_from_original,V)

def class_face(k, test_from_mean, test_flat_images, V, substract_mean_from_original, train_labels):
    eigen_weights = np.dot(V[:k, :], substract_mean_from_original.T)
    threshold = 6000
    for i in range(test_from_mean.shape[0]):
        test_weight = np.dot(V[:k, :], test_from_mean[i:i + 1, :].T)
        distances_euclidian = np.sum((eigen_weights - test_weight) ** 2, axis=0)
        image_closest = np.argmin(np.sqrt(distances_euclidian))
        classification = train_labels[image_closest]
        fig, axes_array = plt.subplots(1, 2)
        fig.set_size_inches(5, 5)
        to_plot = np.reshape(test_flat_images[i, :], (425, 425))
        #axes_array[0].imshow(to_plot, cmap=plt.cm.gray)
        #axes_array[0].axis('off')
        #if (distances_euclidian[image_closest] <= threshold):
        #    axes_array[1].imshow(face_array[image_closest, :, :], cmap=plt.cm.gray)
        #axes_array[1].axis('off')
        return classification
    #plt.show()


def returning_vector(test_images):
    flat_test_Array = []
    for x in range(len(test_images)):
        flat_Array = test_images[x].flatten()
        flat_test_Array.append(flat_Array)
    flat_test_Array = np.asarray(flat_test_Array)
    return flat_test_Array


def reading_test_images():
    # Convert it to jpg
    count = 0
    for image in os.listdir('Test'):
        img_path = 'Test/' + image
        img = cv2.imread(img_path)

        os.remove(img_path)
        cv2.imwrite("Test/"+"im" + str(count) + ".jpg", img)
        count += 1
    # ==========================

    test_images = []
    for images in glob.glob('Test/*.jpg'):  # assuming jpg
        test_faces = Image.open(images)
        res = face_detection(np.asarray(test_faces))
        if res == -1:
            continue
        test_faces = Image.fromarray(res[0])
        test_faces = test_faces.resize((425, 425)).convert('L')
        test_faces = np.asarray(test_faces, dtype=float) / 255.0
        test = (425, 425, 3)
        if test_faces.shape == test:
            test_faces = test_faces[:, :, 0]
            test_images.append(test_faces)
        else:
            test_images.append(test_faces)
    print(len(test_images))
    flat_test_Array = returning_vector(test_images)
    test_images = np.asarray(test_images)
    return flat_test_Array, test_images

def Train():
    face_array, train_labels = reading_faces_and_displaying()
    mean, flatten_Array = performing_pca(face_array)  # eigen_values,eigen_vectors

    substract_mean_from_original = np.subtract(flatten_Array, mean)
    _, _, V = np.linalg.svd(substract_mean_from_original, full_matrices=False)
    Eigen_faces = []
    for x in range(V.shape[0]):
        fig = np.reshape(V[x], (425, 425))
        Eigen_faces.append(fig)

    k = 25
    reconstructing_faces(k, mean, substract_mean_from_original, V)
    write_file("V.csv", V)
    write_file("original.csv", substract_mean_from_original)

    file = open("labels.csv", "w")
    csvwriter = csv.writer(file)
    for i in range(len(train_labels)):
        row = [train_labels[i]]
        csvwriter.writerow(row)
    file.close()

    file = open("mean.csv", "w")
    csvwriter = csv.writer(file)
    for i in range(len(mean)):
        row = [mean[i]]
        csvwriter.writerow(row)
    file.close()

    print(mean.shape)


def Read_File(name):

    file = open(name, "r")
    lines = file.readlines()
    V = []

    for l in lines:
        line = l.split(',')
        temp = []
        for i in range(len(line)):
            toint = float(line[i])
            temp.append(toint)
        V.append(temp)

    file.close()

    return np.asarray(V)

def Read_Data():
    substract_mean_from_original = Read_File("original.csv")
    V = Read_File("V.csv")

    train_labels = []
    file = open("labels.csv", "r")
    lines = file.readlines()
    for l in lines:
        train_labels.append(l)

    mean = []
    file = open("mean.csv", "r")
    lines = file.readlines()
    for l in lines:
        mean.append(float(l))

    return np.asarray(V),np.asarray(substract_mean_from_original),np.asarray(mean),np.asarray(train_labels)

def Recognize(image,V, substract_mean_from_original, mean, train_labels ):

    test_images = []

    test_faces = Image.fromarray(image)
    test_faces = test_faces.resize((425, 425)).convert('L')
    test_faces = np.asarray(test_faces, dtype=float) / 255.0
    test = (425, 425, 3)
    if test_faces.shape == test:
        test_faces = test_faces[:, :, 0]
        test_images.append(test_faces)
    else:
        test_images.append(test_faces)

    test_flat_images = returning_vector(test_images)

    test_images = np.asarray(test_images)

    test_from_mean = np.subtract(test_flat_images, mean)

    k = 25

    return class_face(k, test_from_mean, test_flat_images, V, substract_mean_from_original, train_labels)

# Train()

V, substract_mean_from_original, mean, train_labels = Read_Data()
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    res = face_detection2(img)

    if res != -1:
        img2 = res[0]
        face = res[1]
        print(face)
        (x, y, w, h) = face
        # if ( w*h < 30000):
        #    continue
        #c = test_img(img2, 1)
        name = Recognize(img2,V, substract_mean_from_original, mean, train_labels)
        # print(w*h)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img = cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


# img = cv2.imread("1.jpeg")
# img = face_detection(img)
# if img != -1:
#     img = img[0]    
#     # cv2.imwrite("res.jpg",img)
#     name = Recognize(img,V, substract_mean_from_original, mean, train_labels)
#     print(name)
# else:
#     print("sorry :(")


