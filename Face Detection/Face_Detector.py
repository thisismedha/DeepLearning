import cv2

# Read Trained Data
trainedData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read image 
# inputImg = cv2.imread('msoffice.jpg')
# or
# open your web camera for input; 0 for webcamp any other arguement for a video
webcam = cv2.VideoCapture(0)

# Loop through the frames 
while True:

    #Read current frame
    success, frame = webcam.read()

    # Change the frame to greyscale
    inputFrameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Face Coordinates
    faceCoordinates = trainedData.detectMultiScale(inputFrameGrey)

    # Draw rectangle around the faces
    for (x, y , w, h) in faceCoordinates:
        cv2.rectangle(frame,(x, y ), (x+w, y+h), (0, 255, 0), 2 )

    # Display image
    cv2.imshow('Face Detector', frame)
    # Change frame every 1ms
    key = cv2.waitKey(1)

    # Stop when Q is pressed (ASCII value)
    if key == 81 or key == 113:
        break

# Close Webcam
webcam.release()

print('Successfully Completed')
