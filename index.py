import cv2
import numpy as np
import glob

original = cv2.imread("original_golden_bridge.jpg")

sift = cv2.xfeatures2d.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(original, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

all_images_to_compare = []

titles = []

for f in glob.iglob('./images/*'):
    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)

for image_to_compare, title in zip(all_images_to_compare, titles):
    # 1) check if 2 images are equals
    if original.shape == image_to_compare.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(r) == 0 and cv2.countNonZero(r) == 0:
            print("similarity: 100% (equal size and channels)")
    # 2) check for similarities between the 2 images
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Title: " + title)
    percentage_similarity = len(good_points) / number_keypoints * 100
    print("Similarity " + str(int(percentage_similarity)) + "%\n")

    if percentage_similarity > 90:
        gp = sorted(good_points, key=lambda x: x.distance)
        matching_result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, gp[:50], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matching_result = cv2.resize(matching_result, (960, 640))
        cv2.imshow("Matching: " + title, matching_result)

cv2.waitKey(0)
cv2.destroyWindow()
