import cv2
import mediapipe as mp
import csv
import os

def collect_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    cap = cv2.VideoCapture(0)
    
    csv_file = 'dataset.csv'
    # Hlavička súboru: X0, Y0, ..., X20, Y20, Label
    header = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y']] + ['label']
    
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
            
        print("Stlač číslo (napr. 1, 2, 3) pre uloženie gesta, 'q' pre koniec.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                # Získame všetkých 42 súradníc do jedného listu
                data = [coord for lm in landmarks.landmark for coord in [lm.x, lm.y]]
                
                # Zobrazenie inštrukcií
                cv2.imshow("Data Collector", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif ord('1') <= key <= ord('9'):
                    label = chr(key)
                    writer.writerow(data + [label])
                    print(f"Uložené gesto: {label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()