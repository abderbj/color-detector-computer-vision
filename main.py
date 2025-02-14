import cv2
from util import get_limits

yellow = [0, 255, 255]  # Yellow in BGR colorspace
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Get the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding box

        # Draw the rectangle on the frame
        if w * h > 500:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
