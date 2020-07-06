import glob
import os
import pickle

import cv2

import normalize_img


def start():
    init_output_folder()
    for path in get_all_video_path():
        name = get_file_name_from_path(path)
        video = cv2.VideoCapture(path)
        extract_and_save(name, video)
        video.release()




def extract_and_save(name, video):
    fps = get_fps(video)
    skip_frame_count = get_skip(fps)
    sift = cv2.xfeatures2d.SIFT_create()
    init_output_video_features_folder(name)
    current_frame = 0
    skip_frame = 0
    #
    while True:
        ret, frame = video.read()
        current_frame += 1
        if skip_frame > 0:
            skip_frame -= 1
            continue
        else:
            skip_frame = skip_frame_count
        print(current_frame)
        if ret:
            frame = normalize_img.normalize_video_image(frame)
            kp, desc = sift.detectAndCompute(frame, None)
            index = []
            for point in kp:
                temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
                index.append(temp)
            save_feature(name, fps, current_frame, index, desc)
        else:
            break


def save_feature(name, fps, current_frame, kps, descs):
    f_name = name + '_' + str(current_frame) + '.data'
    f_obj = open('db/' + name + '/' + f_name, 'wb+')
    data = {
        "name": name,
        "fps": fps,
        "frame": current_frame,
        "keypoints": kps,
        "descriptors": descs
    }
    pickle.dump(data, f_obj)
    f_obj.close()


def get_fps(video):
    return video.get(cv2.CAP_PROP_FPS)


def get_skip(fps):
    return int(fps / 3)


def get_file_name_from_path(path):
    return path.split('/')[-1]


def get_all_video_path():
    try:
        if not os.path.exists('input'):
            os.makedirs('input')
    except OSError:
        print("Error: Creating directory of input")
    return glob.iglob("input/*.mp4")


def init_output_folder():
    try:
        if not os.path.exists('output'):
            os.makedirs('output')
    except OSError:
        print("Error: Creating directory of output")


def init_output_video_features_folder(name):
    path = 'db/' + name
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Creating directory of " + path)


start()
