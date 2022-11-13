import math
import cv2
import numpy as np

class FeatureExtract:
    def __init__(self, name):
        self.name = name
        self.detector = self.InitDetector(name)
        self.matcher = self.InitMatcher(name)

    def InitMatcher(self, name):
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_LSH    = 6        
        chunks = name.split('-')
        if 'flann' in chunks and self.norm != None:
            if self.norm == cv2.NORM_L2:
                flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:
                flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
            self.match = "flann"
            return cv2.FlannBasedMatcher(flann_params, {})
        else:
            self.match = "bf"
            return cv2.BFMatcher(self.norm)

    def Detect(self, image):
        kp, des = self.detector.detectAndCompute(image, None)
        if type(des) == type(None):
            return ([],[],[])
        kpToReturn = []
        for singleKp in kp:
            kp_temp = (singleKp.pt, singleKp.size, singleKp.angle, singleKp.response, singleKp.octave, singleKp.class_id)
            kpToReturn.append(kp_temp)
        return (des, kpToReturn, kp)

    def InitDetector(self, name):
        chunks = name.split('-')
        if chunks[0] == 'sift':
            detector = cv2.SIFT_create(800)
            self.norm = cv2.NORM_L2
        elif chunks[0] == 'surf':
            detector = cv2.xfeatures2d.SURF_create(800)
            self.norm = cv2.NORM_L2
        elif chunks[0] == 'orb':
            detector = cv2.ORB_create(nfeatures=1000)
            self.norm = cv2.NORM_HAMMING
        elif chunks[0] == 'akaze':
            detector = cv2.AKAZE_create(threshold=0.00005, nOctaves=2, nOctaveLayers=2, descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)
            self.norm = cv2.NORM_HAMMING
        elif chunks[0] == 'brisk':
            detector = cv2.BRISK_create()
            self.norm = cv2.NORM_HAMMING
        elif chunks[0] == 'fast':
            detector = cv2.FastFeatureDetector_create()
            self.norm = cv2.NORM_L2
        else:
            return None
        
        return detector

    def ransacFilter(self, matches, kp1, kp2):
        newMatches = [x for x in matches if len(x) >= 2]
        good = []
        M = None
        matchesMask = []
        for m, n in newMatches:
            if m.distance < 0.75 *n.distance:
                good.append(m)
        if len(good) > 5: # 10: TODO
            src_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
            matchesMask = mask.ravel().tolist()
        if type(M) != type(None):
            if self.filterHomography(M):
                return matchesMask, good
        return [], []

    def filterHomography(self, homography):
        N3 = math.sqrt(homography[2][0] * homography[2][0] + homography[2][1] * homography[2][1])
        if N3 > 0.002:
            return False
        
        return True