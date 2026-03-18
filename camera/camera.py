import cv2 as cv


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def get_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# When everything is done, release the capture
def release():
    cap.release()
    cv.destroyAllWindows()
