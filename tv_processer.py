import numpy as np
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.draw import circle_perimeter
from skimage.color import gray2rgb


class TVProcesser():
    def __init__(self, image):
        self.image = image

    def calculate_curvature(self, skeleton, window_size=5, curvature_threshold=0.2):
        labeled_skeleton = label(skeleton)
        curvature_acute_points = []
        curvature_obtuse_points = []

        for region in regionprops(labeled_skeleton):
            coords = region.coords

            for i in range(window_size, len(coords) - window_size):
                prev_point = coords[i - window_size]
                current_point = coords[i]
                next_point = coords[i + window_size]

                # Calculate vectors
                vector_1 = np.array(current_point) - np.array(prev_point)
                vector_2 = np.array(next_point) - np.array(current_point)
                
                # Normalize vectors
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                
                # Calculate the angle between the two vectors
                dot_product = np.dot(unit_vector_1, unit_vector_2)

                if dot_product > 0:  # Acute angle
                    curvature_acute_points.append(current_point)
                else:  # Obtuse angle
                    angle = np.arccos(max(dot_product, -1))
                    if angle > curvature_threshold:
                        curvature_obtuse_points.append(current_point)

        return np.array(curvature_acute_points), np.array(curvature_obtuse_points)

    def mark_curvature_points(self, points, radius=5, color=(255, 0, 0)):
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            marked_image = gray2rgb(self.image)
        else:
            marked_image = self.image.copy()

        for point in tqdm(points, desc="Marking points"):
            rr, cc = circle_perimeter(point[0], point[1], radius=radius, shape=marked_image.shape[:2])
            marked_image[rr, cc] = color
            
        return marked_image