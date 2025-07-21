import cv2
import numpy as np

def read_images(idx, path):
    """
    Reads a pair of (stereo) images, given an index and path.
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(path + 'image_0\\' + img_name, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path + 'image_1\\' + img_name, cv2.IMREAD_GRAYSCALE)
    return img1, img2


def compute_kps_descs_matches(img1, img2, threshold=0.002, cross_check=True):
    # Choosing Accelerated-KAZE algorithm, utilizing both performance and accuracy
    akaze = cv2.AKAZE_create(threshold=threshold)

    # Detect keypoints and compute descriptors for both images
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # Because of the binary character of each des1[i], des2[j] - it's natural to choose Hamming Distance
    # Which Xors them and then sums up the Xored elemented.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, des1, kp2, des2, matches

def read_cameras(path):
    with open(path + "calib.txt") as f:
        l1 = f.readline().split()[1:]          # skip first token
        l2 = f.readline().split()[1:]          # skip first token

    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)

    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)

    k  = m1[:, :3]                             # intrinsic matrix
    m1 = np.linalg.inv(k) @ m1                 # extrinsic (left)
    m2 = np.linalg.inv(k) @ m2                 # extrinsic (right)

    return k, m1, m2
