import numpy as np
import cv2

class Panaroma:
    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=4.0, match_status=False):
        (imageB, imageA) = images
        kA, fA = self.detect_feature_and_keypoints(imageA)
        kB, fB = self.detect_feature_and_keypoints(imageB)

        vals = self.match_keypoints(kA, kB, fA, fB, lowe_ratio, max_Threshold)
        if vals is None:
            return None, None if match_status else None

        matches, H, status = vals
        result = self.get_warp_perspective(imageA, imageB, H)
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        if match_status:
            vis = self.draw_matches(imageA, imageB, kA, kB, matches, status)
            return result, vis
        return result

    def get_warp_perspective(self, imageA, imageB, H):
        w = imageA.shape[1] + imageB.shape[1]
        return cv2.warpPerspective(imageA, H, (w, imageA.shape[0]))

    def detect_feature_and_keypoints(self, image):
        sift = cv2.SIFT_create()
        kps, feats = sift.detectAndCompute(image, None)
        return np.float32([kp.pt for kp in kps]), feats

    def match_keypoints(self, kA, kB, fA, fB, lowe_ratio, max_Threshold):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw = matcher.knnMatch(fA, fB, 2)
        valid = [(m[0].trainIdx, m[0].queryIdx)
                 for m in raw if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio]
        if len(valid) <= 4:
            return None
        ptsA = np.float32([kA[i] for (_, i) in valid])
        ptsB = np.float32([kB[i] for (i, _) in valid])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, max_Threshold)
        return valid, H, status

    def draw_matches(self, imgA, imgB, kA, kB, matches, status):
        (hA, wA) = imgA.shape[:2]
        vis = np.zeros((max(hA, imgB.shape[0]), wA + imgB.shape[1], 3), dtype="uint8")
        vis[0:hA, 0:wA] = imgA
        vis[0:imgB.shape[0], wA:] = imgB
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s:
                ptA = (int(kA[queryIdx][0]), int(kA[queryIdx][1]))
                ptB = (int(kB[trainIdx][0]) + wA, int(kB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis
