import sys
import cv2
import os
from tqdm import tqdm


def dissect_video(file_path, dump_folder):
    # if len(sys.argv) == 1:
    #     file_path = input(
    #         "please enter the path to the video to be dismembered:\n")
    # else:
    #     file_path = sys.argv[1]

    capture = cv2.VideoCapture(file_path)
    try:
        video_name = os.path.basename(file_path)[:-4]

        # dump_dir = os.path.join('.', os.path.basename(os.path.dirname(file_path)), video_name + '_dump')
        dump_dir = dump_folder
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = dump_dir + '/{:05}.png'
        print(file_path, dump_dir, dump_path)

        idx = 0
        frames = 0
        while capture.grab():
            frames += 1
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # progress_bar = tqdm(total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        progress_bar = tqdm(total=frames)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            cv2.imwrite(dump_path.format(idx), frame)
            idx += 1
            # if not idx%500:
            #     progress_bar.write(idx)
            progress_bar.update()
    finally:
        capture.release()
    return dump_dir
