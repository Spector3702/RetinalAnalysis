import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from common import compute_save_path, read_ptl_gray_image, bitwise_and_mask, plot_image


def erase_bv(gray_org_img, gray_bv_img, inpaint_radius=3):
    inpaint_mask = cv2.bitwise_not(np.array(gray_bv_img))
    inpainted_img = cv2.inpaint(np.array(gray_org_img), inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted_img


def circle_mask(gray_org_img, radius=500):
    image_center = (gray_org_img.size[0] // 2, gray_org_img.size[1] // 2)
    retina_mask = np.zeros_like(gray_org_img, dtype=np.uint8)
    cv2.circle(retina_mask, image_center, radius, 255, -1)
    return retina_mask


def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    return enhanced_img


def inpaint_small_cnts(img, size_threshold=3200):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < size_threshold:
            cv2.drawContours(img, [cnt], -1, (0), thickness=cv2.FILLED)

    return img


def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY


def cluster_largest_cnts(contours, epsilon, min_samples):
    centers = np.array([get_contour_center(c) for c in contours])
    areas = np.array([cv2.contourArea(c) for c in contours])

    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(centers)
    labels = clustering.labels_

    cluster_areas = {}
    for label, area in zip(labels, areas):
        if label != -1:  # Ignore outliers
            if label in cluster_areas:
                cluster_areas[label] += area
            else:
                cluster_areas[label] = area

    if not cluster_areas:
        print("No clusters found. All contours are treated as outliers. Consider increasing 'epsilon'.")
        return []

    largest_cluster_label = max(cluster_areas, key=cluster_areas.get)
    filtered_contours = [contour for contour, label in zip(contours, labels) if label == largest_cluster_label]

    return filtered_contours


def draw_convex_hull_of_contours(image, contours):
    all_points = np.vstack(contours).squeeze()
    hull = cv2.convexHull(all_points)
    
    hull_img = image.copy()
    cv2.drawContours(hull_img, [hull], -1, (255), 10)

    return hull_img


def circle_csr(org_path, bv_path):
    org_img, gray_org_img = read_ptl_gray_image(org_path)
    bv_img, gray_bv_img = read_ptl_gray_image(bv_path)

    inpainted_img = erase_bv(gray_org_img, gray_bv_img)

    enhanced_img = enhance_contrast(inpainted_img)
    retina_mask = circle_mask(gray_org_img)
    filtered_circle_img = cv2.bitwise_and(enhanced_img, retina_mask)

    csr_area_mask = bitwise_and_mask(filtered_circle_img, 40, 80)
    csr_area_mask = csr_area_mask.astype(np.uint8)
    blurred_image = cv2.GaussianBlur(csr_area_mask, (27, 27), 0)

    inpaint_img = inpaint_small_cnts(blurred_image)
    contours, _ = cv2.findContours(inpaint_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = cluster_largest_cnts(contours, 300, 5)
    
    hull_img = draw_convex_hull_of_contours(np.array(org_img), filtered_contours)
    
    save_path = compute_save_path(bv_path, 'circle')
    original_with_contours = cv2.cvtColor(hull_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, original_with_contours)


if __name__ == '__main__':
    original_path = 'data/Sample_imgs/CSR/814.png'
    blood_vessel_path = 'data/Sample_imgs/CSR_processed/814_bloodvessel.png'
    circle_csr(original_path, blood_vessel_path)