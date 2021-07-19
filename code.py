import cv2
import numpy as np
from scipy import ndimage as ndi
import os   
    
    

frame = cv2.imread("./Imgs/00001.png")

bbox = cv2.selectROI(frame, False)

#print ("bbox: ",bbox)

frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

startingPoint = [bbox[0],bbox[1]]
endPoint = [bbox[0]+bbox[2],bbox[1]+bbox[3]]

x1 = startingPoint[0]
y1 = startingPoint[1]

x2 = endPoint[0]
y2 = endPoint[1]
    

#print ("starting point: ",startingPoint)
#print ("ending point: ",endPoint)

prev_img = frame1

for imgs in os.listdir("./Imgs/"):

    image = cv2.imread("./Imgs/"+imgs)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dy, dx = np.gradient(image)

    Ixx = sum(sum(ndi.gaussian_filter(dx**2, sigma=1)))
    Ixy = sum(sum(ndi.gaussian_filter(dy*dx, sigma=1)))
    Iyy = sum(sum(ndi.gaussian_filter(dy**2, sigma=1)))

    dt = prev_img - image

    Ixt = sum(sum(ndi.gaussian_filter(dx*dt, sigma=1)))
    Iyt = sum(sum(ndi.gaussian_filter(dy*dt, sigma=1)))

    inv = np.linalg.inv([[Ixx,Ixy],[Ixy,Iyy]])

    #print ("inv: ",inv)

    t = np.zeros((2,1))

    t[0] = -Ixt
    t[1] = -Iyt

    p = np.dot(inv,t)

    prev_img = image

    #print ("sp: ",[x1,y1])
    #print ("ep: ",[x2,y2])

    #print ("p: ",p)

    x1 = x1 + p[0][0]
    y1 = y1 + p[1][0]
    x2 = x2 + p[0][0]
    y2 = y2 + p[1][0]

    #print ("sp: ",[x1,y1])
    #print ("ep: ",[x2,y2])

    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,255), 2)

    cv2.imshow("image", image)

    cv2.waitKey(0)

    #cv2.imwrite("./output/"+imgs, image)

