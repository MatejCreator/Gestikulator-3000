import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    # max_num_hands=1 keeps it fast and focused on one hand for shortcuts
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    # Open the default webcam (0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("KRITICKÁ CHYBA: Nepodarilo sa pripojiť ku kamere! Skontroluj index alebo práva vo Windowse.")
        return

    print("Starting Gestikulator 3000. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        # Flip the frame horizontally for a more natural "mirror" feel
        frame = cv2.flip(frame, 1)
        
        # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Draw the 21 landmarks if a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

        # Display the output window
        cv2.imshow("Gestikulator 3000 - Vision Core", frame)

        # Break the loop and close the app if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and release the camera hardware
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()