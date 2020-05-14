import glob
import json
import os
import pickle

import cv2
import numpy

import change_fps
from json import JSONEncoder

import normalize_img


def save_result(img1, img2, kp1, kp2, good_points, name):
    name = './output/frame_' + name + '.jpg'
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, good_points[:200], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(name, matching_result)


def find_object_in_video(object_img, video):
    # object_img = normalize_img.normalize_object_image(object_img)
    fps = change_fps.get_fps(video)
    # init algorithm
    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm=0, tree=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #
    kp1, desc_1 = sift.detectAndCompute(object_img, None)
    #
    current_frame = 0
    skip_frame = 0
    # run
    while True:
        ret, frame = video.read()
        current_frame += 1
        if ret:
            if skip_frame != 0:
                skip_frame -= 1
                continue
            #
            # frame = normalize_img.normalize_video_image(frame)
            kp2, desc_2 = sift.detectAndCompute(frame, None)
            percentage, good_points = check_similarity(flann, 0.7, kp1, desc_1, kp2, desc_2)
            print("Match: " + str(percentage) + "%")
            if percentage > 20.:
                t = current_frame / fps
                save_result(object_img, frame, kp1, kp2, good_points,
                            str(round(t, 2)) + "s_" + str(round(percentage, 2)) + "%")
            else:
                skip_frame = int(fps / 2)
        else:
            break


def check_similarity(flann, threshold, kp1, desc_1, kp2, desc_2):
    good_points = []
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_points.append(m)
    if len(kp1) > len(kp2):
        n_keypoints = len(kp2)
    else:
        n_keypoints = len(kp1)

    percentage_similarity = len(good_points) / n_keypoints * 100
    return percentage_similarity, good_points


def start():
    obj_img = cv2.imread('input/theme2.png')
    obj_img = normalize_img.normalize_object_image(obj_img)
    detect(obj_img)
    # initData()


def initData():
    try:
        if not os.path.exists('input'):
            os.makedirs('input')
    except OSError:
        print("Error: Creating directory of input")
    #
    for f in get_all_video_file_name():
        f_name = f.split('/')[-1]
        v = cv2.VideoCapture(f)
        extract_and_save_f(f_name, v)


def detect(img):
    try:
        if not os.path.exists('input'):
            os.makedirs('input')
    except OSError:
        print("Error: Creating directory of input")
    try:
        if not os.path.exists('input'):
            os.makedirs('input')
    except OSError:
        print("Error: Creating directory of input")
    #
    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm=0, tree=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for f in get_all_data_file_name():
        data = get_data(f)
        detect_img_in_vid(img, data, sift, flann)


def detect_img_in_vid(img, data, sift, flann):
    kp1, desc_1 = sift.detectAndCompute(img, None)
    name = data["name"]
    fps = data["fps"]
    for p in data["data"]:
        frame = p["frame"]
        raw_kps = p["keypoints"]
        raw_desc = p["descriptors"]

        kp2 = []
        desc_2 = raw_desc
        for r_kp in raw_kps:
            kp = json_to_keypoint(r_kp)
            kp2.append(kp)
        matches = flann.knnMatch(desc_1, desc_2, k=2)
        good_points = []

        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
        if len(kp1) > len(kp2):
            n_keypoints = len(kp2)
        else:
            n_keypoints = len(kp1)
        percentage_similarity = len(good_points) / n_keypoints * 100

        if percentage_similarity > 20.:
            print("tìm thấy ảnh đầu vào trong: " + name)
            print("xuất hiện : " + str(round(frame / fps, 2)) + "s")


def json_to_keypoint(point):
    return cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                        _octave=point[4], _class_id=point[5])


def json_to_descriptor(point):
    return numpy.asarray(point)


def get_data(f_name):
    return pickle.load(open(f_name, 'rb'))


def get_all_data_file_name():
    return glob.iglob("input/*.data")


def get_all_video_file_name():
    return glob.iglob("input/*.mp4")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_skip(fps):
    return int(fps / 3)


def extract_and_save_f(v_name, video):
    f_obj = open('input/' + v_name + '.data', 'wb+')
    fps = change_fps.get_fps(video)
    skip = get_skip(fps)
    # init algorithm
    sift = cv2.xfeatures2d.SIFT_create()
    current_frame = 0
    skip_frame = 0
    datas = []
    # run
    while True:
        ret, frame = video.read()
        current_frame += 1
        if skip_frame > 0:
            skip_frame -= 1
            continue
        else:
            skip_frame = skip
        print(current_frame)
        if ret:
            frame = normalize_img.normalize_video_image(frame)
            kp, desc = sift.detectAndCompute(frame, None)
            index = []
            for point in kp:
                temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
                index.append(temp)
            data = {
                "frame": current_frame,
                "keypoints": index,
                "descriptors": desc
            }
            datas.append(data)
        else:
            break

    pickle.dump({
        "name": v_name,
        "fps": fps,
        "data": datas
    }, f_obj)


start()
