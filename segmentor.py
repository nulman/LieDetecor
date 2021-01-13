import re
from bisect import bisect_left
from collections import defaultdict

# from DBFace.common import imread

# import face_recognition
# import numpy as np
import cv2
from skimage import io
import dlib
from glob import glob
import os
from os import path
from video_dissector import dissect_video
from db import DB

from skimage.metrics import structural_similarity
from tqdm.contrib.concurrent import process_map
# from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from functools import lru_cache
# from scipy.spatial.distance import cdist
import pickle
import csv
import colorama
from ast import literal_eval
import traceback
from functools import partial
from itertools import combinations


# import torch
# from DBFace.main import detect
# from DBFace.model.DBFace import DBFace
# from DBFace.common import imread


# from random import randint

class RangeList(list):
    def index(self, __value, __start: int = 0, __stop: int = -1) -> int:
        if __stop == -1:
            __stop = len(self)
        for i, item in enumerate(self[__start:__stop]):
            if __value in item:
                return i + __start
        else:
            tqdm.write(f"{colorama.Fore.RED}-E- {__value} not found {colorama.Fore.RESET}")
            raise ValueError("Value not present")


class Application:
    # image_dir = r'GB1.02\VTS_01_4_dump'  # TODO: make this an input parameter
    # image_dir = None
    # filtered_dir = os.path.join(os.path.dirname(image_dir), 'filtered_' + os.path.basename(image_dir))
    # filtered_dir = None

    def __init__(self, image_dir, filtered_dir):
        self.image_dir = image_dir
        self.filtered_dir = filtered_dir
        self.revealed_scene = False
        self.template = cv2.imread('pattern.png')  # pattern of viewport separator in split screen scenes
        self.scene_ranges = None
        # _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(img, self.template, cv2.TM_CCOEFF_NORMED))

    # os.makedirs(filtered_dir, exist_ok=True)

    @staticmethod
    def overlap(rect1, rect2):
        # bottom, right, top, left
        # 'low y, high x, high y, low x'
        if rect1[3] >= rect2[1] or rect2[3] >= rect1[1]:
            return False
        if rect1[2] <= rect2[0] or rect2[2] <= rect1[0]:
            return False
        return True

    @staticmethod
    @lru_cache(maxsize=None)
    def get_detector():
        """
        initializes and returns a frontal face detector.
        due to caching, actual initialization happens only **once** per (sub)process
        :return: dlib frontal face detector
        """
        print("initializing detector...")
        '''# detector = dlib.get_frontal_face_detector()
        detector = partial(face_recognition.face_locations, model='cnn')
        # detector = None
        caspath = path.join(cv2.data.haarcascades, r"haarcascade_profileface.xml")
        side = cv2.CascadeClassifier(caspath)
        # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        return detector, side  # , predictor'''
        caspath = path.join(cv2.data.haarcascades, r"haarcascade_profileface.xml")
        side = cv2.CascadeClassifier(caspath)
        from DBFace.DBFace import DBF
        import face_recognition
        dbface = DBF()
        # dbface.eval()
        # if torch.cuda.is_available():
        #     dbface.cuda()
        # dbface.load("DBFace/model/dbface.pth")
        return dbface.detect, side, face_recognition

    def filter_faces(self, file):
        detector, side_face_detector, face_recognition = self.get_detector()
        # im = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # im = imread(file)
        # face_rects, _, _ = detector.run(im, 2, 0.3)
        # if '00126' in path.basename(file):
        #     tqdm.write(f'in {file}')
        # face_rects = detector(im)
        # face_rects = [list(map(int, item.box)) for item in face_rects]
        face_rects = [list(map(int, [item.y, item.x + item.width, item.y + item.height, item.x])) for item in
                      detector(im) if item.area > 400]
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # face_rects = face_recognition.face_locations(im, model='cnn')
        # predictor(im, face_rects[0][0])
        box = face_rects
        # if len(face_rects) > 0:
        #     # new_path = path.join(self.filtered_dir, path.basename(file))
        #     if len(face_rects) == 1:
        #         box.append(self.get_box(face_rects[0]))
        #     if len(face_rects) > 1:
        #         box.extend([self.get_box(rect) for rect in face_rects])
        #     # os.renames(file, new_path)
        # print(box)
        for item in map(self.get_side_box, side_face_detector.detectMultiScale(
                im, scaleFactor=1.1, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)):
            box.append(item)
        # print(box)
        if len(box) > 1:
            fset = set()
            skip = None
            for a, b in combinations(box, 2):
                a = tuple(a)
                b = tuple(b)
                if skip is a:
                    continue
                fset.add(b)
                if self.overlap(a, b):
                    if a in fset:
                        fset.remove(a)
                    skip = a
                else:
                    fset.add(a)
            box = list(fset)
        # print(box)
        # print('-'*80)
        # if '00126' in path.basename(file):
        #     tqdm.write(f'{face_recognition.face_encodings(im, known_face_locations=box, num_jitters=3)}, {path}, {box}')
        if len(box):
            try:
                # self.show_face(file, box=box, multiple=True)
                img_enc = face_recognition.face_encodings(im, known_face_locations=box, num_jitters=3)
                return {path.basename(file): [{'encoding': enc, 'rect': rect} for enc, rect in zip(img_enc, box)]}
            except TypeError as e:
                print(e)
                print(im.shape, box)

    # def get_image_encodings(self, fp):
    #     img = cv2.imread(fp)
    #     detector = self.get_detector()
    #     faces = [self.get_box(face) for face in detector.run(img, 1, 0.4)[0]]
    #     img_enc = face_recognition.face_encodings(img, known_face_locations=faces, num_jitters=3)
    #     return img_enc

    @staticmethod
    def get_box(rect):
        # return rect.top(), rect.right(), rect.bottom(), rect.left()
        # return rect[1], rect[0], rect[3] - rect[1], rect[2]-rect[0]
        return rect

    @staticmethod
    def get_side_box(rect):
        return rect[1], rect[0] + rect[2], rect[1] + rect[3], rect[0]

    def list_scenes(self):
        files = glob(os.path.join(self.image_dir, '*.png'))
        pkl_fp = os.path.join(self.filtered_dir, 'ssim.pkl')
        if os.path.exists(pkl_fp):
            with open(pkl_fp, 'rb') as fh:
                mapping = pickle.load(fh)
        else:
            mapping = self.calculate_image_diff(files)
            with open(pkl_fp, 'wb') as fh:
                pickle.dump(mapping, fh)
        results = []
        for file in files:
            key = path.basename(file)
            if key not in mapping:
                continue
            value = mapping[key]
            # if value < 0.44:
            if value < 0.55:
                print(key, value)
                results.append(key)
        if len(results) == 0:
            raise Exception(f"no results, is mapping empty? files:{len(files)}, mapping:{len(mapping)}")
        end = path.basename(files[-1])
        results[-1] = str(int(end[:-4]) + 1) + end[-4:]
        self.scene_ranges = RangeList(
            [range(int(results[i][:-4]), int(results[i + 1][:-4])) for i in range(len(results) - 1)])
        return self.scene_ranges

    # TODO: this is unused
    # @staticmethod
    # def get_dom_colors(file):
    #     """
    #     :param file: path to image
    #     :return: int(image name), top 4 dominant colors
    #     """
    #     n_colors = 4
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    #     flags = cv2.KMEANS_RANDOM_CENTERS
    #     img = io.imread(file)
    #     pixels = np.float32(img.reshape(-1, 3))
    #
    #     _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 100, flags)
    #     _, counts = np.unique(labels, return_counts=True)
    #     colors = palette[np.argsort(counts)[:n_colors - 1]]
    #     return int(os.path.basename(file).split('.')[0]), colors

    @staticmethod
    def similarity(images):
        img1 = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(images[1], cv2.IMREAD_GRAYSCALE)
        # return {path.basename(images[1]): structural_similarity(img1, img2, multichannel=True)}
        return {path.basename(images[1]): structural_similarity(img1, img2)}

    def calculate_image_diff(self, files):
        """
        calculates image similarity score
        :param files: path list of all the image (frame) files
        :return: dictionary mapping each frame to the previous frame
        """
        ssim = {}
        prev = None
        collection = [[files[i], files[i + 1]] for i in range(len(files) - 1)]
        print("generating image differences for scene segmentation...")
        for item in process_map(self.similarity, collection, chunksize=3):
            if item is not None:
                print(item)
                ssim.update(item)
        ssim.update({'00000.png': 0.0})
        # for file in tqdm(files):
        #     img = cv2.imread(file)
        #     if prev is None:
        #         ssim[path.basename(file)] = 0
        #     else:
        #         ssim[path.basename(file)] = structural_similarity(prev, img, multichannel=True)
        #     prev = img
        return ssim

    def get_contestant_encoding(self, name, listing: dict, n_images=3):
        # listing = {'file.png': [face1, face2, ...]}
        # print(list(listing.keys())[0])
        suffix = '.png'
        numerical = [int(item[:-4]) for item in listing.keys()]
        numerical.sort()
        raw_input = ''
        while not raw_input:
            raw_input = input(f"please provide comma or space separated gender and {n_images} for {name} > ")
        split_input = re.split(r'[,\s]+', raw_input)
        res = [0 if split_input.pop(0).strip().lower().startswith('m') else 1]
        idx = 0
        while idx < len(split_input):
            if split_input[idx] in listing:
                idx += 1
            else:
                frame_num = int(split_input[idx][:-4])
                for i in range(len(self.scene_ranges)):
                    detected_frames = [detected for detected in self.scene_ranges[i] if detected in numerical]
                    if frame_num in detected_frames:
                        print(f'{colorama.Fore.YELLOW}{split_input[idx]} has no face in the '
                              f'scene try one of the following:({[f"{num:05}.png" for num in detected_frames]})'
                              f'{colorama.Fore.RESET}')
                        break
                    elif frame_num < self.scene_ranges[i].start:
                        print(f'{colorama.Fore.YELLOW}{frame_num:05}.png is not in a scene '
                              f'with faces{colorama.Fore.RESET}')
                        break
                tidx = bisect_left(numerical, frame_num)
                subs = ''
                if tidx > 0:
                    subs += f'{numerical[tidx - 1]:05}{suffix} and '
                subs += f'{numerical[tidx]:05}{suffix}'

                split_input[idx] = input(f"(closest frames are {subs}), please input again > ")
        for ans in split_input:
            for key in listing:
                if ans in key:
                    face = listing[key][0]
                    if len(listing[key]) > 1:
                        print(f"-W- more than one face found, please select the correct face")
                        face = listing[key][self.show_face(key, [item['rect'] for item in listing[key]], True)]
                    elif len(listing[key]) == 0:
                        print(f"-E- {len(listing[key])} faces found. Expected at least one face in the image")
                        break
                    res.append(face['encoding'])
                    break
            else:
                print(f"-E- image {ans} was not found, skipping")
        return res

    def show_face(self, im_path, box=None, multiple=False, hold=False, title='Image'):
        if type(im_path) is str:
            img = cv2.imread(im_path)
            if img is None:
                img = cv2.imread(path.join(self.image_dir, im_path))
            print(im_path)
        else:
            img = im_path
        if multiple and box is not None:
            for i, item in enumerate(box):
                cv2.rectangle(img, (item[1], item[0]), (item[3], item[2]), (0, 255, 0), 2)
                print('show:', item)
                # cv2.rectangle(img, item, (0, 255, 0), 2)
                cv2.putText(img, f'{i}', (item[1] + 5, item[0] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        elif box is not None:
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
        cv2.imshow(title, img)
        try:
            key = cv2.waitKey(0) & 0xFF
            if box is not None:
                return int(chr(key))
            else:
                return chr(key)
        except ValueError:
            cv2.destroyAllWindows()
            return
        finally:
            if not hold:
                cv2.destroyAllWindows()

    def map_scene_type(self, scenes, paths):
        reveal = False
        result = {id: 0 for id in range(len(scenes))}
        print(f"press {colorama.Fore.YELLOW}0{colorama.Fore.RESET} for scenes "
              f"{colorama.Fore.YELLOW}before{colorama.Fore.RESET} the reveal")
        print(f"press {colorama.Fore.YELLOW}1{colorama.Fore.RESET} for scenes "
              f"{colorama.Fore.YELLOW}during{colorama.Fore.RESET} the reveal")
        print(f"press {colorama.Fore.YELLOW}2{colorama.Fore.RESET} for scenes "
              f"{colorama.Fore.YELLOW}after{colorama.Fore.RESET} the reveal")
        print(f'press {colorama.Fore.RED}q{colorama.Fore.RESET} to end')
        print(f'press any other key to skip frame')

        image_path_mapping = {int(loc[:-4]): loc for loc in paths}
        try:
            title = 'img:{} - 0 before, 1 during, 2 after, 3 interview, 4 other'
            for i, scene in list(enumerate(scenes)):
                last = None
                for frame in scene:
                    if frame in image_path_mapping:
                        last = image_path_mapping[frame]
                        if not self.revealed_scene:
                            img = cv2.imread(last)
                            if img is None:
                                img = cv2.imread(path.join(self.image_dir, last))
                            _, max_val, _, max_loc = cv2.minMaxLoc(
                                cv2.matchTemplate(
                                    img, self.template, cv2.TM_CCOEFF_NORMED))
                            if max_val > 0.8:
                                self.revealed_scene = True
                                reveal = True
                                result[i] = 1
                            else:
                                result[i] = 0
                            print(f"{path.basename(last)} : {result[i]}")
                            break
                        elif reveal:
                            img = cv2.imread(last)
                            if img is None:
                                img = cv2.imread(path.join(self.image_dir, last))
                            _, max_val, _, max_loc = cv2.minMaxLoc(
                                cv2.matchTemplate(
                                    img, self.template, cv2.TM_CCOEFF_NORMED))
                            if max_val > 0.8:
                                result[i] = 1
                                print(f"{path.basename(last)} is still : {result[i]}")
                                break
                            else:
                                reveal = False

                        category = self.show_face(last, title=title.format(path.basename(last)))
                        if category in ['0', '1', '2', '3', '4']:
                            result[i] = int(category)
                            break
                        elif category in ['q', 'Q']:
                            return result
                    elif last is None:
                        result[i] = 4
                    else:
                        while True:
                            print("please tag this scene #")
                            category = self.show_face(last, hold=True, title=title.format(path.basename(last)))
                            if category in ['0', '1', '2', '3', '4']:
                                result[i] = int(category)
                                break
            return result
        finally:
            cv2.destroyAllWindows()

    def sql_to_csv(self, handle=None):
        headers = 'video contestant_1_Decision contestant_2_Decision scene start_frame end_frame scene_type ' \
                  'contestant gender location_x location_y ' \
                  'contestant gender location_x location_y ' \
                  'contestant gender location_x location_y'.split(' ')
        # results = [headers]
        row_stringefier = lambda x: [str(item) for item in x]
        with open(path.join(self.filtered_dir, 'csv2.csv'), 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(headers)
            if handle:
                db = handle
            else:
                db = DB(self.filtered_dir)
            scenes = db.select('select distinct scene from data')
            decisions = db.select('select decision from decision order by  contestant asc')
            for scene in scenes:
                scene_error_tolerance = db.select(f'select count(DISTINCT frame) from data where scene == {scene}')[0] // 2.5
                error_test = db.select(f'select contestant, count(contestant) from data where '
                                       f'scene == {scene} group by contestant')
                real_contestant = tuple(item[0] for item in error_test if item[1] > scene_error_tolerance)
                # real_contestant = tuple(
                #     [item[0] for item in error_test if item[1] > max(error_test, key=lambda x: x[1])[1] / 5])
                if len(real_contestant) == 1:
                    real_contestant = f'({real_contestant[0]})'

                selector = db.select("select printf('%05d', min(frame)), "
                                     "printf('%05d', max(frame)), "
                                     'scene_type '
                                     f'from data where scene == {scene}')
                faces = db.select(f'select facepoints, contestant, gender from data where scene == {scene}'
                                  f' and contestant in {real_contestant} group by contestant order by  contestant asc')
                contestants = []
                for row in faces:
                    points = literal_eval(row[0])
                    contestants.append(row[1])
                    contestants.append(row[2])
                    contestants.append((points[1] + points[3]) // 2)
                    contestants.append((points[0] + points[2]) // 2)
                contestants += [''] * (12 - len(contestants))
                file_name = path.basename(path.dirname(self.image_dir))
                row = [file_name] + decisions + [scene] + selector[0] + contestants
                print(row)
                # if ',' in row[3]:
                #     row_genders = []
                #     for contestant in row[3].split(','):
                #         row_genders.append(str(db.select(f'select distinct(gender) from data where contestant == {contestant}')))
                #     row[4] = ','.join(row_genders)
                # results.append(row_stringefier(row))
                writer.writerow(row_stringefier(row))


class SqlAssistant:
    def __init__(self, contestant_info):
        self.contestant_info = contestant_info

    def sql_entry(self, frame_num, participant_index, face, scene_id):
        return [f'{frame_num:05}',
                str(face),
                self.contestant_info[participant_index]['gender'],
                scene_id,
                self.contestant_info[participant_index]['id'],
                scene_types[scene_id]
                ]

    def entry(self, frame_id, participant_index, face, scene_id):
        return [frame_id,
                self.contestant_info[participant_index]['id'],
                self.contestant_info[participant_index]['gender'],
                face,
                scene_id]


if __name__ == '__main__':
    # sql_to_csv()
    # exit()
    # image_path = r'E:\Workspace\hagit\LieDetector\GB1.01\VTS_01_4.VOB_dump'
    image_path = r'E:\Workspace\hagit\LieDetector\GB1.02\VTS_01_4_dump'
    # dissect_video(r"E:\Workspace\hagit\GB1.02\GB1.02\VTS_01_4.VOB.mpg", image_path)
    app = Application(image_path, os.path.join(os.path.dirname(image_path), 'filtered_' + os.path.basename(image_path)))
    print(f'image dir:{app.image_dir}\nfiltered dir:{app.filtered_dir}')
    # app.image_dir = dissect_video(r"E:\Workspace\hagit\GB1.02\GB1.02\VTS_01_4.VOB.mpg")
    # app.image_dir = r'E:\Workspace\hagit\LieDetector\GB1.01\VTS_01_4.VOB_dump'
    # app.filtered_dir = os.path.join(os.path.dirname(app.image_dir), 'filtered_' + os.path.basename(app.image_dir))
    os.makedirs(app.filtered_dir, exist_ok=True)
    print(f'image dir:{app.image_dir}\nfiltered dir:{app.filtered_dir}')

    samples = 3
    results = {}
    sqldata = []
    try:
        if not len(results):
            with open(path.join(app.filtered_dir, 'encodings.pkl'), 'rb') as fh:
                results = pickle.load(fh)
    except Exception:
        print('generating new encodings...')
    if not results:
        nons = 0
        images = glob(os.path.join(app.image_dir, r'*.png'))
        # images = [(item, app.filtered_dir) for item in images]
        # NOTE: lower max_workers if memory is an issue
        for item in process_map(app.filter_faces, images, chunksize=10, max_workers=3):
        # for impath in tqdm(images):
        #     item = app.filter_faces(impath)
            if item is not None:
                results.update(item)
            else:
                nons += 1
        print(nons)
        if len(results) > 0:
            with open(path.join(app.filtered_dir, 'encodings.pkl'), 'wb') as fh:
                pickle.dump({k: v for k, v in results.items() if v is not None}, fh)
    scene_ranges = app.list_scenes()
    print(scene_ranges)
    scene_types = app.map_scene_type(scene_ranges, results)
    print(scene_types)

    # TODO: prompt for results
    print(len(results))
    c1 = app.get_contestant_encoding('the host', results, n_images=3)
    c1_gen = c1.pop(0)
    c2 = app.get_contestant_encoding('contestant one', results, n_images=3)
    c2_gen = c2.pop(0)
    c3 = app.get_contestant_encoding('contestant two', results, n_images=3)
    c3_gen = c3.pop(0)

    participants = [*c1, *c2, *c3]
    genders = [c1_gen, c2_gen, c3_gen]
    contestant_info = [{'id': 0, 'gender': c1_gen},
                       {'id': 1, 'gender': c2_gen},
                       {'id': 2, 'gender': c3_gen}]
    # face_recognition.compare_faces(thing[142][1], thing[143][1][0])
    tagged = []
    skipped = 0
    queries = SqlAssistant(contestant_info)
    # entry = lambda frame_id, participant_index, face, scene_id: [frame_id, contestant_info[participant_index]
    # ['id'], contestant_info[participant_index]['gender'], face, scene_id]
    #
    # sql_entry = lambda frame_num, participant_index, face, scene_id: [f'{frame_num:05}',
    #                                                                   str(face),
    #                                                                   contestant_info[participant_index]['gender'],
    #                                                                   scene_id,
    #                                                                   contestant_info[participant_index]['id'],
    #                                                                   scene_types[scene_id]
    #                                                                   ]
    # entry = lambda frame_id, participant_index, face,scene_id: f'''{frame_id}, {contestant_info[participant_index]
    #     ['id']}, {contestant_info[participant_index]['gender']}, {face}, {scene_id}'''
    import face_recognition

    for frame, data in tqdm(results.items()):
        for i, face in enumerate(data):
            threshold = 0.6
            ruling = False
            while True:
                face_rulings = defaultdict(list)
                detection = face_recognition.compare_faces(participants, face['encoding'], threshold)
                face_rulings[sum(detection[:samples])].append(0)
                face_rulings[sum(detection[samples:2 * samples])].append(1)
                face_rulings[sum(detection[2 * samples:])].append(2)
                likely_face = max(face_rulings)
                if len(face_rulings[likely_face]) > 1:
                    threshold -= 0.05
                    if threshold < 0.35:
                        break
                    continue
                elif len(face_rulings[likely_face]) < 1:
                    threshold += 0.05
                    if threshold > 0.8:
                        break
                ruling = True
                break
            if not ruling:
                tqdm.write(f"{colorama.Fore.RED}-E- {frame} ({likely_face}) doesnt match {face_rulings[likely_face]}! "
                           f"with threshold:{threshold} skipping...{colorama.Fore.RESET}")
                # show_face(frame, face['rect'])
                skipped += 1
                continue
            elif threshold != 0.6:
                tqdm.write(f"-I- {frame} ({likely_face}) matches {face_rulings[likely_face]} "
                           f"with threshold:{threshold}")
            tagged.append(queries.entry(path.basename(frame), face_rulings[likely_face][0], face['rect'], 0))
            frame_num = int(path.basename(frame)[:-4])
            sqldata.append(
                queries.sql_entry(frame_num, face_rulings[likely_face][0], face['rect'], scene_ranges.index(frame_num)))

    print(f"-I- Skipped {skipped} faces")
    decisions = []
    while len(decisions) != 2:
        print(f"did player {colorama.Fore.RED}{len(decisions) + 1}{colorama.Fore.RESET} "
              f"split(0) or steal(1)?")
        dec = input('>').lower()
        if dec.startswith('0') or dec.startswith('sp'):
            decisions.append([str(len(decisions) + 1), '0'])
        elif dec.startswith('1') or dec.startswith('st'):
            decisions.append([str(len(decisions) + 1), '1'])
        else:
            print('-E- please enter either 0/1 or split/steal')
    # with open(path.join(filtered_dir, 'csv.pkl'), 'wb') as fh:
    #     pickle.dump(tagged, fh)
    # with open(path.join(filtered_dir, 'csv.csv'), 'w') as fh:
    #     writer = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    #     writer.writerows(tagged)
    db = DB(app.filtered_dir, overwrite=True)
    for row in sqldata[:5]:
        for col in row:
            print(col, ':', type(col), ', ', end='')
        print('')
    try:
        db.insert(sqldata)
        db.insert_decisions(decisions)
        app.sql_to_csv(db)
    except Exception:
        traceback.print_exc()
        print(f'{colorama.Fore.RED}data dumped to dump.txt{colorama.Fore.RESET}')
        with open('dump.txt', 'w') as fh:
            for item in sqldata:
                print(item, file=fh)
