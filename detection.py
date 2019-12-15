import cv2
import numpy as np
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

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw
def imageEnhance(img):
    r=img[:,:,2]
    g=img[:,:,1]

    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    y=ycrcb[:,:,0]

    ymax=np.max(y)
    ymin=np.min(y)

    y=255*((y-ymin)/(ymax-ymin))
    yavg=np.average(y)
    print (yavg)
    T=1
    if yavg<64:
        T=1.4
    if yavg>192:
        T=0.6
    newimg=np.copy(img)
    Rnew=np.power(r,T)
    Gnew=np.power(g,T)
    newimg[:,:,2]=Rnew
    newimg[:,:,1]=Gnew

    r=newimg[:,:,2]
    g=newimg[:,:,1]
    b=newimg[:,:,0]

    ycbcr=cv2.cvtColor(newimg,cv2.COLOR_BGR2YCR_CB)
    y=ycrcb[:,:,0]
    cr=ycrcb[:,:,1]
    cb=ycrcb[:,:,2]

    hsv=cv2.cvtColor(newimg,cv2.COLOR_BGR2HSV)
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]


    mask=(r>150) & (g>60) & (b>100) & (abs(r-g)>=30)  & (r>g) & (r>b)
    mask2=(cb>=90) & (cb<=135)& (cr>140) & (cr<165)
    mask3=((h>0) & (h<35/2)) | ((h>300/2) & (h<360/2)) & (s/255>0.2) &(s/255<0.6)
    skin=np.zeros((len(img),len(img[0])))
    skin[(mask2) & (mask3)]=1
    r[skin==0]=0
    g[skin==0]=0
    b[skin==0]=0

    # cv2.imshow("skin", skin)

    newimg=cv2.medianBlur(newimg,21)
    gray=cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    gray[gray>0]=255


    cv2.imshow("skin", gray)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    skin = cv2.dilate(gray, kernel, iterations=3)





    output = cv2.connectedComponentsWithStats(skin, 4)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids=output[3]


    for i in range(1, labels.max() + 1):
        width=stats[i,cv2.CC_STAT_WIDTH]
        height=stats[i,cv2.CC_STAT_HEIGHT]
        area=stats[i,cv2.CC_STAT_AREA]
        areRec=width*height
        x,y=centroids[i]
        minorAxis=x+width/2
        majorAxis=y+height/2
        ecc=minorAxis/majorAxis
        ratArea = area / areRec
        ratio=width/height
        pts = np.where(labels == i)
        print(len(pts[0]))
        if len(pts[0]) < 1500:
            labels[pts] = 0

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    # cv2.imshow("new2", labeled_img)

    gray = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    gray[gray > 0] = 255
    gray = cv2.erode(gray, kernel, iterations=1)


    output = cv2.connectedComponentsWithStats(gray, 4)
    num_labels = labels[0]
    labels = output[1]
    stats = output[2]

    area = 0
    index = 0
    for i in range(1, labels.max() + 1):
        if stats[i, cv2.CC_STAT_AREA] > area:
            area = stats[i, cv2.CC_STAT_AREA]
            index = i

    x = stats[index, cv2.CC_STAT_LEFT]
    y = stats[index, cv2.CC_STAT_TOP]
    w = stats[index, cv2.CC_STAT_WIDTH]
    h = stats[index, cv2.CC_STAT_HEIGHT]
    #cv2.rectangle(img, (x,- y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("sd",labeled_img)
    side=max(w,h)
    return x,y,side,side

# img=cv2.imread("im1.jpg")
# #img=cv2.resize(img,(500,500))
# imageEnhance(img)


