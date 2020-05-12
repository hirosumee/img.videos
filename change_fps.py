import cv2


def getFps(video):
    return video.get(cv2.CAP_PROP_FPS)


def enoughFps(video):
    fps = getFps(video)
    if fps > 10.0:
        return False
    return True


