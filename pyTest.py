import cv2


capture = cv2.VideoCapture("C:/Users/Wolf/Desktop/ALPR/Video/sampleVideo1.mp4")

while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("Video/sampleVideo1.mp4")

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(33) > 0: break

capture.release()
cv2.destroyAllWindows()

# image = cv2.imread("C:/Users/Wolf/Desktop/ALPR/Image/overstack.jpg", cv2.IMREAD_UNCHANGED)
# cv2.imshow("Stack", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
