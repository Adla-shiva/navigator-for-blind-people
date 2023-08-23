import cv2

# read video
cap = cv2.VideoCapture(0)

# initialize background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2()

# initialize list to store object speeds
object_speeds = []

while True:
    # read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # apply background subtraction
    fg_mask = back_sub.apply(frame)
    
    # apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # find contours of objects in foreground
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over contours and calculate object speeds
    for contour in contours:
        # get bounding box of object
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # filter out small objects
        if w * h < 2000:
            continue
        
        # filter out objects that are too high or too low
        if y < 100 or y > 500:
            continue
        
        # calculate object speed based on previous frame
        if len(object_speeds) > 0:
            last_x, last_y, last_w, last_h, last_speed = object_speeds[-1]
            curr_speed = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5 / (1.0 / cap.get(cv2.CAP_PROP_FPS))
        else:
            curr_speed = 0.0
        
        # store object speed
        object_speeds.append((x, y, w, h, curr_speed))
        
        # draw bounding box and speed on frame
        if w * h > 10000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{curr_speed:.2f} px/s', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # show frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
