import cv2
import easyocr
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

reader_gpu = easyocr.Reader(['en'], gpu=True)  
video_path_gpu = 'video path'
output_path_gpu = 'output path'
cap_gpu = cv2.VideoCapture(video_path_gpu)

fps_gpu = cap_gpu.get(cv2.CAP_PROP_FPS)
total_frames_gpu = int(cap_gpu.get(cv2.CAP_PROP_FRAME_COUNT))
width_gpu = int(cap_gpu.get(cv2.CAP_PROP_FRAME_WIDTH))
height_gpu = int(cap_gpu.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_gpu = cv2.VideoWriter(output_path_gpu, fourcc, int(fps_gpu), (width_gpu, height_gpu))
frame_count_gpu = 0
fps_list_gpu = []
t1_gpu = time.time()

while True:
    ret, frame = cap_gpu.read()
    if not ret:
        break
    frame_count_gpu += 1
    start_time = time.time()
    ocr_results_gpu = reader_gpu.readtext(frame)
    end_time = time.time()
    fps_list_gpu.append(1 / (end_time - start_time))
    for (bbox, text, prob) in ocr_results_gpu:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out_gpu.write(frame)
cap_gpu.release()
out_gpu.release()


avg_fps_gpu = sum(fps_list_gpu) / len(fps_list_gpu)
print(f"GPU processing complete. Avg FPS: {avg_fps_gpu}")

#### CPU ####

reader_cpu = easyocr.Reader(['en'], gpu=False)  
video_path_cpu = 'vid2.mp4'
output_path_cpu = 'output_cpu.mp4'
cap_cpu = cv2.VideoCapture(video_path_cpu)

fps_cpu = cap_cpu.get(cv2.CAP_PROP_FPS)
total_frames_cpu = int(cap_cpu.get(cv2.CAP_PROP_FRAME_COUNT))
width_cpu = int(cap_cpu.get(cv2.CAP_PROP_FRAME_WIDTH))
height_cpu = int(cap_cpu.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_cpu = cv2.VideoWriter(output_path_cpu, fourcc, int(fps_cpu), (width_cpu // 2, height_cpu // 2))
frame_count_cpu = 0
fps_list_cpu = []
t1_cpu = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    while True:
        ret, frame = cap_cpu.read()
        if not ret:
            break

        frame_count_cpu += 1
        if frame_count_cpu % 2 != 0:
            continue
        
        start_time = time.time()
        frame_resized = cv2.resize(frame, (width_cpu // 2, height_cpu // 2))
        ocr_results_cpu = reader_cpu.readtext(frame_resized)
        end_time = time.time()
        fps_list_cpu.append(1 / (end_time - start_time))
        for (bbox, text, prob) in ocr_results_cpu:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(frame_resized, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame_resized, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        out_cpu.write(frame_resized)

cap_cpu.release()
out_cpu.release()

avg_fps_cpu = sum(fps_list_cpu) / len(fps_list_cpu)

print(f"CPU processing complete. Avg FPS: {avg_fps_cpu}")

frames_gpu = list(range(1, len(fps_list_gpu) + 1))
frames_cpu = list(range(1, len(fps_list_cpu) + 1))
print(len(frames_cpu),len(frames_gpu))

plt.figure(figsize=(10, 5))
plt.plot(frames_gpu, fps_list_gpu, label="GPU FPS", color='blue', marker='o')
plt.plot(frames_cpu, fps_list_cpu, label="CPU FPS", color='green', marker='x')
plt.xlabel("Frame Number")
plt.ylabel("Frames Per Second (FPS)")
plt.title("FPS Comparison: GPU vs CPU")
plt.legend()
plt.grid(True)
plt.show()

print(f"Average FPS (GPU): {avg_fps_gpu}")
print(f"Average FPS (CPU): {avg_fps_cpu}")

