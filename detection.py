import cv2
import numpy as np
import math
#from numba import jit, cuda
from skimage.color import rgb2hsv, rgb2ycbcr


def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """

    if n not in (4, 8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows, cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows, cols))
    south = np.zeros((rows, cols))
    west = np.zeros((rows, cols))
    east = np.zeros((rows, cols))

    north[:-1, :] = bw[1:, :]
    south[1:, :] = bw[:-1, :]
    west[:, :-1] = bw[:, 1:]
    east[:, 1:] = bw[:, :-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west == bw) & \
          (east == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:] = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:] = bw[:-1, :-1]
        south_west[1:, :-1] = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw


def Histogram(image):
    histo = np.zeros(256)
    for i in range(len(image)):
        for j in range(len(image[0])):
            histo[image[i, j]] += 1

    return histo


def getImageWithHist(img, nbins=256):
    gray = img
    h = Histogram(gray)
    H_c = np.zeros(256)
    H_c[0] = h[0]
    En = np.copy(gray)
    for i in range(1, len(h)):
        H_c[i] = H_c[i - 1] + h[i]

    q = np.round((nbins - 1) * H_c / (len(gray) * len(gray[0])))
    for i in range(len(gray)):
        for j in range(len(gray[0])):
            En[i, j] = q[gray[i, j]]
    return En


def segment(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('detected circles', img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) * 0.564 + 128
    Cr = (R - Y) * 0.713 + 128

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    Skin = np.zeros((len(img), len(img[0])))
    length = len(img)
    width = len(img[0])
    size = length * width

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    EyeMapC = ((1 / 3) * (
                (Cb ** 2 / 255).astype(np.uint8) + (((255 - Cr) ** 2) / 255).astype(np.uint8) + (Cb / Cr).astype(
            np.uint8))).astype(np.uint8)

    # e=getImageWithHist(EyeMapC)

    sigma = np.sqrt(size) / 24

    Cr[Cr < 135] = -1
    Cr[Cr > 135] = 1

    Cb[Cb < 85] = -3
    Cb[Cb > 85] = 1

    h[h >= 0] = 1
    h[h > 50] = -9

    s[s >= 0.1 * 255] = 1
    s[s > 0.9 * 255] = -12

    Ta = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Tb = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    Skin[Cr - Cb - s == -1] = 1

    # Skin=Ta[1] * Tb[1] * 255

    kernel = np.ones((11, 7))
    Skin = cv2.erode(Skin, kernel, iterations=1)
    kernel = np.ones((21, 11))
    Skin = cv2.dilate(Skin, kernel, iterations=2)

    Skin = Skin.astype(np.uint8) * 255

    output = cv2.connectedComponentsWithStats(Skin, 4)

    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    if num_labels > 0:
        area = 0
        index = 0
        for i in range(1, labels.max() + 1):
            if stats[i, cv2.CC_STAT_AREA] > area:
                area = stats[i, cv2.CC_STAT_AREA]
                index = i
        '''
        comps=[]
        for i in range(num_labels):
            comps.append([])
        for i in range(length):
            for j in range(width):
                comps[labels[i,j]].append(gray[i,j])
        stds=[]
        for i in range(1,len(comps)):
            stds.append(np.std(comps[i]))
        '''

        # if area>39000:
        # index=np.argmax(stds)+1
        labels[labels != index] = 0
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0

        x = stats[index, cv2.CC_STAT_LEFT]
        y = stats[index, cv2.CC_STAT_TOP]
        w = stats[index, cv2.CC_STAT_WIDTH]
        h = stats[index, cv2.CC_STAT_HEIGHT]
        edges = cv2.Canny(gray, 100, 200)

        edges[labeled_img[:, :, 0] == 0] = 0
        co = np.copy(img)
        co[labeled_img[:, :, 0] == 0] = 0
        gray = cv2.cvtColor(co, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((21, 21))
        labeled_img = cv2.dilate(labeled_img, kernel, iterations=3)
        # kernel = np.ones((21, 21))
        labeled_img = cv2.erode(labeled_img, kernel, iterations=3)

        c = 0

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=40, param2=29, maxRadius=25)
        dots = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            perW = 0.1
            perH = 0.05
            for i in circles[0, :]:
                if (EyeMapC[i[1] - 1, i[0] - 1] > 30) and (i[1] < y + h / 2):
                    dots.append((i[0], i[1]))
                    c += 1
                    # draw the outer circle
                    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        EyeMapC[labeled_img[:, :, 0] == 0] = 0

        if c > 0:
            mdist = 0
            mindist = 500
            l = 0
            di = (0, 0)
            dj = (0, 0)
            if len(dots) > 1:

                for i in range(len(dots)):
                    for j in range(i + 1, len(dots)):
                        x2 = dots[j][0]
                        x1 = dots[i][0]
                        y2 = dots[j][1]
                        y1 = dots[i][1]
                        dx = int(x2) - int(x1)
                        dy = int(y2) - int(y1)

                        l = math.sqrt((dx) ** 2 + (dy) ** 2)
                        if l < mindist and l > 60 and -30 < dy < 30:
                            di = dots[i]
                            dj = dots[j]
                            mindist = l

                if mindist < 150:
                    cv2.line(img, di, dj, (0, 0, 255), 2)  # line between eyes
                    x = max([min([di[0], dj[0]]) - int(mindist), 0])
                    y = max([min([di[1], dj[1]]) - int(1.5 * mindist), 0])
                    h = int(4 * mindist)
                    w = int(3 * mindist)

                    im = gray[y:y + h, x:x + w]

                    myradians = math.atan2(int(dj[1]) - int(di[1]), int(dj[0]) - int(di[0]))
                    mydegrees = math.degrees(myradians)
                    image_center = tuple(np.array(im.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(image_center, mydegrees, 1.0)
                    rot = cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)

                    flip = cv2.flip(rot, 1)
                    sym = 0
                    for i in range(len(rot)):
                        for j in range(len(rot[0])):
                            if -40 < int(rot[i, j]) - int(flip[i, j]) < 40:
                                sym += 1
                    sym = sym / (len(rot) * len(rot[0])) * 100
                    st = np.std(rot)
                    print(sym, st)

                    if 20 < sym < 90:
                        return x,y,w,h
                        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


    return -1