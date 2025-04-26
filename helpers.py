import cv2, imutils, os

def load_and_resize(path, width=400, height=400):
    img = cv2.imread(path)
    img = imutils.resize(img, width=width)
    return imutils.resize(img, height=height)

def show_and_save(result, matched, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    if matched is not None:
        cv2.imshow("Keypoint Matches", matched)
        cv2.imwrite(f"{outdir}/matched_points.jpg", matched)
    cv2.imshow("Panorama", result)
    cv2.imwrite(f"{outdir}/panorama_image.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
