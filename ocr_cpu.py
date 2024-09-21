
import cv2
import easyocr
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

reader = easyocr.Reader(['en'], gpu=False)
video_path = 'path to video'
output_path = 'path to output'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("error could not open")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, int(fps), (width // 2, height // 2)) # i have decreased the resolution of video for optimization
frame_count = 0
t1 = time.time()

with ThreadPoolExecutor(max_workers=4) as executor: # for optimising the model for cpu i have used multi threading
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}", end='\r')
        if frame_count % 2 != 0:
            continue
      
        frame = cv2.resize(frame, (width // 2, height // 2))
        ocr_results = reader.readtext(frame)
        
        for (bbox, text, prob) in ocr_results:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(round((time.time() - t1)*1000, 4))
print(f"Video processed successfully!")
