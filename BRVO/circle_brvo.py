import os
import cv2
import numpy as np
from PIL import Image
from skimage import util

from common import compute_save_path, read_ptl_gray_image, bitwise_and_mask


def subtract_bv(gray_original_img, gray_blood_vessel_img):
    inverted_blood_vessel_img = util.invert(np.array(gray_blood_vessel_img))
    subtracted_img = np.clip(np.array(gray_original_img, dtype=np.int16) - inverted_blood_vessel_img, 0, 255).astype(np.uint8)
    return subtracted_img


def distance_from_center(point, center):
    return np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)


def avg(points):
    x, y = zip(*points)
    return sum(x) / len(x), sum(y) / len(y)


def filter_contours_by_anypnts_distance(contours, center, distance_threshold):
    filtered_contours = []
    for contour in contours:
        if all(distance_from_center(pt[0], center) < distance_threshold for pt in contour):
            filtered_contours.append(contour)

    return filtered_contours


def filter_contours_by_area(size_threshold, contours):
    filter_cnts = []

    for cnt in contours:
        if cv2.contourArea(cnt) > size_threshold:
            filter_cnts.append(cnt)

    return filter_cnts


def circle_brvo(original_path, blood_vessel_path):
    original_img, gray_original_img = read_ptl_gray_image(original_path)
    _, gray_blood_vessel_img = read_ptl_gray_image(blood_vessel_path)

    subtracted_img = subtract_bv(gray_original_img, gray_blood_vessel_img)
    brvo_area_mask = bitwise_and_mask(subtracted_img, 10, 89)

    brvo_area_mask_uint8 = brvo_area_mask.astype(np.uint8) * 255
    blurred_image = cv2.GaussianBlur(brvo_area_mask_uint8, (27, 27), 0)
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_center = (original_img.size[0] // 2, original_img.size[1] // 2)
    distance_threshold = 2 * original_img.size[1] // 5
    filtered_contours = filter_contours_by_anypnts_distance(contours, image_center, distance_threshold)
    filtered_cnts = filter_contours_by_area(3200, filtered_contours)

    original_with_contours = np.array(original_img).copy()
    cv2.drawContours(original_with_contours, filtered_cnts, -1, (0, 255, 0), 2)

    save_path = compute_save_path(blood_vessel_path, 'circle')
    original_with_contours = cv2.cvtColor(original_with_contours, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, original_with_contours)


if __name__ == '__main__':
    org_dir = 'data/Sample_imgs/BRVO'
    processed_bv_dir = 'data/Sample_imgs/BRVO_processed'

    for org_img_filename in os.listdir(org_dir):
        original_path = os.path.join(org_dir, org_img_filename)
        
        bv_img_filename = org_img_filename.split('.')[0] + '_bloodvessel.png'
        blood_vessel_path = os.path.join(processed_bv_dir, bv_img_filename)

        if os.path.exists(blood_vessel_path):
            circle_brvo(original_path, blood_vessel_path)
        else:
            print(f"No corresponding blood vessel image found for {org_img_filename}")