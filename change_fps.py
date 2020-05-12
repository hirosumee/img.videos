import cv2


def get_fps(video):
    return video.get(cv2.CAP_PROP_FPS)


def enough_fps(video):
    fps = get_fps(video)
    if fps > 10.0:
        return False
    return True


