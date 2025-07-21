import cv2
import matplotlib.pyplot as plt
import os

# --- Magic Constants ---
DATA_PATH = r'C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\sequences\05\\'
LINES_NUM = 20
RATIO_THRESH = 0.70
LINES_NUM_SIG_TEST = 20
LINES_NUM_CORRECT_BAD = 1


def read_images(idx, data_path):
    """
    Reads a pair of (stereo) images, given an index and path.
    """
    img_name = '{:06d}.png'.format(idx)
    img1_path = os.path.join(data_path, "image_0", img_name)
    img2_path = os.path.join(data_path, "image_1", img_name)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one or both images from the specified path.")

    return img1, img2


def display_images(img1, img2):
    """
    Not necessarily used.
    Displays both images: img1, img2.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("Image 2")
    plt.axis("off")

    plt.show()


def draw_keypoints_both_images(img1, kp1, img2, kp2):
    """
    Draws keypoints on a copy of img1, img2
    """
    # --- Creating the corresponding image, including the keypoints ---
    out_img1 = cv2.drawKeypoints(
        img1, kp1, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    out_img2 = cv2.drawKeypoints(
        img2, kp2, None, color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    # --- Displaying the keypoints on each image ---
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.imshow(out_img1)
    plt.title("Keypoints in Image 1")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(out_img2)
    plt.title("Keypoints in Image 2")
    plt.axis("off")

    plt.show()


def draw_matches(matches, img1, kp1, img2, kp2, lines_num, title):
    """
    Draws 'lines_num' matches and specifies a title to the figure.
    """
    # Creating the stereo-pair image, including matches
    result_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:lines_num],
    None,    # create a new image to draw all the matches
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # don't draw keypoints that do not have matches
    )

    # Drawing it
    plt.figure(figsize=(10, 5))
    plt.imshow(result_img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def significance_test(ratio_thresh, knn_matches):
    """
    Performing a significance test, keeping only matches that don't exceed the ratio.
    """
    good_matches = []
    bad_matches = []

    for m, n in knn_matches:
        # m is the best match, n is the second-best match
        if m.distance < (ratio_thresh * n.distance):
            good_matches.append(m)
        else:
            bad_matches.append(m)

    return good_matches, bad_matches


def main(data_path=DATA_PATH,
         lines_num=LINES_NUM,
         ratio_thresh=RATIO_THRESH,
         lines_num_sig_test=LINES_NUM_SIG_TEST,
         lines_num_correct_bad=LINES_NUM_CORRECT_BAD):
    # --- Reading Images ---
    img1, img2 = read_images(0, data_path)

    # Optional; to inspect the uploaded images
    # display_images(img1, img2)

    # --- Detecting Keypoints & Computing Descriptors ---
    akaze = cv2.AKAZE_create(threshold=0.002)   # Accelerated-KAZE algorithm
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # --- Printing Number of Keypoints In Both Images ---
    print(f"Number of keypoints in image1: {len(kp1)}")
    print(f"Number of keypoints in image2: {len(kp2)}")

    # --- Drawing & Displaying Keypoints In Both Images ---
    draw_keypoints_both_images(img1, kp1, img2, kp2)
    print(f"The length of each keypoint descriptor is: {len(des1[0])}")
    print(f"The type of each element within this descriptor is: {des1[0].dtype}, meaning 1 byte\n")
    print(f"Although it seems each element is int, it is actually a binary representation of {8 * len(des1[0])} bits")

    # --- Matching Keypoints & Drawing Matches ---
    # Because of the binary character of each des1[i], des2[j] - it's natural to choose Hamming Distance
    # which Xors them and then sums up the Xored elements.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    title = "Stereo Pair Presenting 20 Matching Lines"
    draw_matches(matches, img1, kp1, img2, kp2, lines_num, title)

    # --- Significance Test ---
    # crossCheck should be False, because we want the two best neighbors for each descriptor
    bf_sig_test = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # kNN match to get two best matches for each descriptor in des1
    knn_matches = bf_sig_test.knnMatch(des1, des2, k=2)

    good_matches, bad_matches = significance_test(ratio_thresh, knn_matches)

    print(f"Total matches before ratio test: {len(knn_matches)}")
    print(f"Matches passing ratio test: {len(good_matches)}")

    # Sort good_matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    title_sig_test = f"Matches after Ratio Test (ratio={ratio_thresh})"
    draw_matches(good_matches, img1, kp1, img2, kp2, lines_num_sig_test, title_sig_test)

    # Sort bad_matches by distance
    bad_matches = sorted(bad_matches, key=lambda x: x.distance)

    title_correct_bad = "A correct match that was nevertheless labeled as 'bad'"
    draw_matches(bad_matches, img1, kp1, img2, kp2, lines_num_correct_bad, title_correct_bad)


if __name__ == "__main__":
    main()
