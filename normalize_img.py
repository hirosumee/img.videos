import cv2


def normalize_video_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # percent by which the image is resized
    # scale_percent = 50
    # calculate the 50 percent of original dimensions
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dsize
    # dsize = (width, height)
    # resize image
    # return img
    # return cv2.resize(img, dsize)


def normalize_object_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
