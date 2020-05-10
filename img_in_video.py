import cv2


def draw(img1, img2, kp1, kp2, gp, percentage):
    good_points = sorted(gp, key=lambda x: x.distance)
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, good_points[:200], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matching_result = cv2.resize(matching_result, (960, 640))
    cv2.imshow("Matching: " + str(percentage) + " %", matching_result)


def compareTwoImage():
    original = cv2.imread("original_golden_bridge.jpg")
    compare_to = cv2.imread("./images/black_and_white.jpg")
    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    percentage, kp1, kp2, good_points = checkSimilarity(sift, flann, 0.6, original, compare_to)
    draw(original, compare_to, kp1, kp2, good_points, percentage)


def checkEquals(img, compare_to):
    if img.shape == compare_to.shape:
        difference = cv2.subtract(img, compare_to)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) and cv2.countNonZero(r):
            return True

    return False


def checkSimilarity(sift, flann, threshold, img, compare_to):
    kp1, desc_1 = sift.detectAndCompute(img, None)
    kp2, desc_2 = sift.detectAndCompute(compare_to, None)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []

    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_points.append(m)

    if len(kp1) > len(kp2):
        n_keypoints = len(kp1)
    else:
        n_keypoints = len(kp2)

    percentage_similarity = len(good_points) / n_keypoints * 100
    return percentage_similarity, kp1, kp2, good_points


compareTwoImage()
cv2.waitKey(0)
cv2.destroyAllWindows()
