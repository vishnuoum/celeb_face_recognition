import os
import cv2
import joblib
import json
import numpy as np
import pywt
import base64



#global variables
face_cascade=None
eye_cascade=None
model=None
celebs,inv_celebs=None,None





#load haarcascade and model
def load():
    global face_cascade,eye_cascade,model,celebs,inv_celebs
    face_cascade=cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")
    model=joblib.load("model/celeb_face_recog_model")
    with open("model/celebs.json",'r') as f:
        celebs=json.loads(f.read())
    inv_celebs={v:k for k,v in celebs.items()}
    return("done")





#fn to detect and return  faces if and only if there are two eyes clearly detected 
def face_and_eye_detect(image):
    roi=[]
    img=cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes)>1):
            roi.append(roi_color)
    return roi




#fn to detect and return  faces if and only if there are two eyes clearly detected for base64 files
def face_and_eye_detect_base64(image):
    # list of faces to return
    roi=[]

    #base64 to cv2 image conversion
    im_bytes = base64.b64decode(image)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes)>1):
            roi.append(roi_color)
    return roi





#fn to convert images to wave
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H






#predict fn
def img_prediction(image):
    load()
    global model
    x=[]
    # imgs=face_and_eye_detect(image)
    imgs=face_and_eye_detect_base64(image)
    for img in imgs:
        r_img=cv2.resize(img,(32,32))
        r_img_har=cv2.resize(w2d(img),(32,32))
        combine=np.vstack((r_img.reshape(32*32*3,1),r_img_har.reshape(32*32,1)))
        x.append(combine)
    #it is to ensure that the dimention of X is n x 4096
    try:
        X=np.array(x).reshape(len(x),4096).astype(float)
        return return_result(model.predict(X),model.predict_proba(X))
    except:
        return "No face detected clearly"




#return the result in a dictionary containig celeb name and confidence
def return_result(x,y):
    result={}
    for i in range(len(x)):
        result[inv_celebs[x[i]]]=np.max(y[i])
    return result




#main method
if(__name__=="__main__"):
    
    
    with open("test/1.jpg", "rb") as f:
        im_b64 = base64.b64encode(f.read())

    


    load()
    print(model)
    print(img_prediction(im_b64))

