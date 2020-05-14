import glob
import pickle

import cv2

import normalize_img


def start():
    obj_img = cv2.imread('input/theme2.png')
    obj_img = normalize_img.normalize_object_image(obj_img)
    #
    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm=0, tree=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    kp1, desc_1 = sift.detectAndCompute(obj_img, None)
    for d_path in get_all_data_file_name():
        data = get_data(d_path)
        detect(flann, kp1, desc_1, data)


def detect(flann, or_kps, or_desc, data):
    name = data["name"]
    fps = data["fps"]
    frame = data["frame"]
    raw_kps = data["keypoints"]
    descs = data["descriptors"]
    kps = to_kps(raw_kps)

    matches = flann.knnMatch(or_desc, descs, k=2)
    good_points = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)
    if len(or_kps) > len(kps):
        n_keypoints = len(kps)
    else:
        n_keypoints = len(or_kps)
    percentage_similarity = len(good_points) / n_keypoints * 100
    if percentage_similarity > 20.:
        print("tìm thấy ảnh đầu vào trong: " + name)
        print("xuất hiện : " + str(round(frame / fps, 2)) + "s")


def to_kps(raw):
    kps = []
    for r_kp in raw:
        kp = json_to_keypoint(r_kp)
        kps.append(kp)
    return kps


def json_to_keypoint(point):
    return cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                        _octave=point[4], _class_id=point[5])


def get_all_data_file_name():
    return glob.iglob("output/**/*.data")


def get_data(f_name):
    f = open(f_name, 'rb')
    data = pickle.load(f)
    f.close()
    return data


start()
