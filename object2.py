# import cv2
# import math
# # importing the pyttsx library
# import pyttsx3
  
# # initialisation
# engine = pyttsx3.init()
  

# # read video
# cap = cv2.VideoCapture(0)

# # initialize background subtractor
# back_sub = cv2.createBackgroundSubtractorMOG2()

# # load haarcascade files for humans, vehicles, and animals
# human_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# vehicle_cascade = cv2.CascadeClassifier('carcascade.xml')
# animal_cascade = cv2.CascadeClassifier('dogharcascade.xml')

# # initialize list to store object speeds
# object_speeds = []

# while True:
#     # read frame
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # apply background subtraction
#     fg_mask = back_sub.apply(frame)
    
#     # apply morphological operations to remove noise
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
#     # detect humans, vehicles, and animals in frame
#     human_rects = human_cascade.detectMultiScale(frame)
#     vehicle_rects = vehicle_cascade.detectMultiScale(frame)
#     animal_rects = animal_cascade.detectMultiScale(frame)
    
#     # loop over detected objects and calculate speeds
#     if(len(human_rects)!=0):
#         for (x, y, w, h) in human_rects:
#         # calculate object speed based on previous frame
#             if len(object_speeds) > 0:
#                 last_x, last_y, last_w, last_h, last_speed = object_speeds[-1]
#                 curr_speed = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5 / (1.0 / cap.get(cv2.CAP_PROP_FPS)) / 1000.0 * 3600.0
#                 object_speeds.append((x, y, w, h, curr_speed))
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 print("the speed of the person is",curr_speed)
#                 engine.say("the speed of the person is ",curr_speed)
#                 cv2.putText(frame, f'{curr_speed:.2f} px/s', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 engine.runAndWait()
    
#             else:
#                 object_speeds.append((x, y, w, h, 0))
          
#     if(len(vehicle_rects)!=0):      
#         for (x, y, w, h) in vehicle_rects:
#         # calculate object speed based on previous frame
#             if len(object_speeds) > 0:
#                 last_x, last_y, last_w, last_h, last_speed = object_speeds[-1]
#                 curr_speed = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5 / (1.0 / cap.get(cv2.CAP_PROP_FPS)) / 1000.0 * 3600.0
#                 object_speeds.append((x, y, w, h, curr_speed))
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{curr_speed:.2f} px/s', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 print("the speed of the vehicle is",curr_speed)
#                 engine.say("the speed of the vehicle is ",curr_speed)
#                 engine.runAndWait()
#             else:
#                 object_speeds.append((x, y, w, h,0))
#     if(len(animal_rects)!=0):       
#         for (x, y, w, h) in animal_rects:
#         # calculate object speed based on previous frame
#             if len(object_speeds) > 0:
#                 last_x, last_y, last_w, last_h, last_speed = object_speeds[-1]
#                 curr_speed = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5 / (1.0 / cap.get(cv2.CAP_PROP_FPS)) / 1000.0 * 3600.0
#                 object_speeds.append((x, y, w, h, curr_speed))
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 print("the speed of the animal is",curr_speed)
#                 engine.say("the speed of the animal is",curr_speed)
#                 cv2.putText(frame, f'{curr_speed:.2f} px/s', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 engine.runAndWait()
    
#             else:
#                 object_speeds.append((x, y, w, h,0))
           
            
            
        
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# # release resources
# cap.release()
# cv2.destroyAllWindows()






# gpt code



import cv2
import numpy as np
import time
import math
import pyttsx3

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load Yolo
net = cv2.dnn.readNet("y olov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set up camera
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    # Detect objects in frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store object information
    class_ids = []
    confidences = []
    boxes = []

    # Extract object information from output layers
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Store object information
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Draw bounding boxes and display object information
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            
            # Calculate distance and speed of object
            distance = round(35 / w, 2) # assuming an average width of 35 cm for detected objects
            elapsed_time = time.time() - starting_time
            speed = round(distance / elapsed_time * 3.6, 2) # convert from m/s to km/hr
            
            # Speak object information
            message = f"{label} at {distance} centimeters and {speed} kilometers per hour"
            engine.say(message)
            engine.runAndWait()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    # Display frame
    # cv2.imshow("Frame", frame)
    # key = cv2.waitKey(1)
    # if key == 27: 
        
        
        # press 'Esc'
cap.release()
cv2.destroyAllWindows()
