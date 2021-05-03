import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20, 60, 80)  # some stupid default

# mouse callback function
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower = np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(image_mask, kernel, iterations=4)
        erosion = cv2.erode(dilation, kernel, iterations=3)

        invert = cv2.bitwise_not(erosion)

        cv2.imshow("invert", invert)

        cv2.imshow("mask", image_mask)

        cv2.imshow("using kernels", erosion)


def main():
    import sys
    global image_hsv, pixel  # so we can use it in mouse callback

    image_src = cv2.imread('RawImages/09.jpg')  # pick.py my.png
    if image_src is None:
        print("no image")
        return
    # cv2.imshow("bgr", image_src)

    # NEW #
    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', pick_color)

    image_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)

    cv2.imshow("hsv", image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
