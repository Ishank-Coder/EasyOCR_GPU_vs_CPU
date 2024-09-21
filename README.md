EasyOCR FPS Optimization for CPU

Step 1: Install Required Packages
Install all necessary dependencies by running

pip install -r requirements.txt

Step 2: Run GPU Version
Execute the ocr_gpu.py file to perform OCR using GPU acceleration. This will help you benchmark the performance of the GPU model:

python ocr_gpu.py

Step 3: Run CPU Version
Execute the ocr_cpu.py file to perform OCR using CPU optimization techniques:

python ocr_cpu.py

Optimization Techniques Used in ocr_cpu.py
Multithreading: Utilizes Python's threading capabilities to process multiple frames simultaneously, improving overall throughput.

Frame Skipping: Skips every other frame (or a specified interval) to reduce the workload, allowing for faster processing without significantly affecting accuracy.

Resolution Reduction: Decreases the video resolution prior to OCR processing. This lowers the computational load and speeds up text recognition, especially useful in real-time applications.
