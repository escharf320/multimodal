import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing from the webcam
cap = cv2.VideoCapture(1)
# video_path = "/Users/eli/Downloads/typing_example2.mov"
# cap = cv2.VideoCapture(video_path)

# print(cv2.getBuildInformation())


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a more natural interaction
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    # Check if hand landmarks are found
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand landmarks
    cv2.imshow("Hand Detection", frame)

    # Print the hand landmarks for debugging
    # print("Hand landmarks:")
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         for id, lm in enumerate(hand_landmarks.landmark):
    #             print(f"Landmark {id}: ({lm.x}, {lm.y}, {lm.z})")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
