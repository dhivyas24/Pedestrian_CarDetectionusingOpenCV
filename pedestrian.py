from random import randrange
import cv2


trained_people = cv2.CascadeClassifier('haarcascade_fullbody.xml')

trained_car = cv2.CascadeClassifier('trained_car.xml')

web = cv2.VideoCapture(0)
while True:
    frameread , imga= web.read()
   
    img = cv2.imread('download.jfif')
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    people = trained_people.detectMultiScale(grayscale)
    car = trained_car.detectMultiScale(grayscale)
    for(x,y,w,h) in people:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    for(xa,ya,wa,ha) in car:
        cv2.rectangle(img,(xa,ya),(xa+wa,ya+ha),(255,255,255),2)

    
    cv2.imshow('Hello',img)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
web.release()

  



