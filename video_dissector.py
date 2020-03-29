import sys
import cv2
import os

if len(sys.argv) == 1:
    file_path = input(
        "please enter the path to the video to be dismembered:\n")
else:
    file_path = sys.argv[1]


capture = cv2.VideoCapture(file_path)
video_name = os.path.basename(file_path)[:-4]
dump_dir = './' + video_name + '_dump'
os.makedirs(dump_dir, exist_ok=True)
dump_path = dump_dir + '/image_{:05}.png'
print(file_path, dump_dir, dump_path)

idx = 0
while True:
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imwrite(dump_path.format(idx), frame)
    idx += 1
    if not idx%500:
        print(idx)
capture.release()