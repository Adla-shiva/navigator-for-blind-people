import cv2  
# This is the distance from camera to face object  
DECLARED_LEN = 30 # cm  
# width of the object face  
DECLARED_WID = 14.3 # cm  
# Definition of the RGB Colors format 
GREEN = (0, 255, 0)  
RED = (255, 0, 0)  
WHITE = (255, 255, 255)
#Defining the fonts family, size, type  
fonts = cv2.FONT_HERSHEY_COMPLEX  
# calling the haarcascade_frontalface_default.xml module for face detection.  
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def focal_length(determined_distance, actual_width, width_in_rf_image):  
    focal_length_value = (width_in_rf_image * determined_distance) / actual_width  
    return focal_length_value


def distance_finder(focal_length, real_face_width, face_width_in_frame):  
    distance = (real_face_width * focal_length) / face_width_in_frame  
    return distance

def face_data(image):  
  face_width = 0  
#   gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  
    #We use 1.3 for less powerful processors but can increase it according to your processing power of your machine.
  faces =face_cascade.detectMultiScale(image,1.05, 5) 
    #getting the rectangular frame 
  for (x, y, h, w) in faces:  
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)  
        face_width = w  
  
  return face_width  

ref_image = cv2.imread("shivaadla.jpg") 
ref_image_face_width = face_data(ref_image) 
#Processing our called reference image
focal_length_found = focal_length(DECLARED_LEN, DECLARED_WID, ref_image_face_width)  
# print(focal_length_found)  
cv2.imshow("ref_image", ref_image)  
cap = cv2.VideoCapture(0)
while True:  
     
    _, frame=cap.read() 
    # calling face_data function  
    face_width_in_frame=face_data(frame)  
    # finding the distance by calling function Distance  
    if face_width_in_frame != 0:  
         Distance=distance_finder(focal_length_found, DECLARED_WID, face_width_in_frame)  
        # Writing Text on the displaying screen  
         cv2.putText(  
            frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (WHITE),2)  
    cv2.imshow('frame', frame)  
    if cv2.waitKey(1) == ord("q"):  
        break  
cap.release()  
cv2.destroyAllWindows()