import os

import cv2
import change_fps
import normalize_img


def save_result(img1, img2, kp1, kp2, good_points, name):
    name = './data/frame_' + name + '.jpg'
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, good_points[:200], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(name, matching_result)


def find_object_in_video(object_img, video):
    object_img = normalize_img.normalize_object_image(object_img)
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
            frame = normalize_img.normalize_video_image(frame)
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
    try:
        if not os.path.exists('data'):
            os.makedirs("data")

    except OSError:
        print("Error: Creating directory of data")
    v = cv2.VideoCapture('./theme_7.mp4')
    v.set(cv2.CAP_PROP_FPS, 10)
    obj_img = cv2.imread('./theme2.png')

    find_object_in_video(obj_img, v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


start()
