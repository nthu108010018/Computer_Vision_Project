import mediapipe as mp
import time
import cv2

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


capture = cv2.VideoCapture(0)

while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()
 
    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))
 
    # Converting the from from BGR to RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
      image,
      results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS,
      mp_drawing.DrawingSpec(
        color=(255,0,255),
        thickness=1,
        circle_radius=1
      ),
      mp_drawing.DrawingSpec(
        color=(0,255,255),
        thickness=1,
        circle_radius=1
      )
    )  

    cv2.imshow("Facial", image)
 
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()