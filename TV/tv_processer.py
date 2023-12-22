import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.draw import circle_perimeter
from skimage.color import gray2rgb


class TVProcesser():
    def __init__(self, image):
        self.image = image

    def get_unit_vectors(self, skeleton, window_size=5):
        labeled_skeleton = label(skeleton)
        unit_vectors = []

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
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1) if np.linalg.norm(vector_1) != 0 else np.array([0,0])
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2) if np.linalg.norm(vector_2) != 0 else np.array([0,0])
                
                unit_vectors.append((unit_vector_1, unit_vector_2, current_point))

        return unit_vectors

    def calculate_curvature_from_vectors(self, unit_vectors, curvature_threshold=0.2):
        curvature_points = []

        for unit_vector_1, unit_vector_2, current_point in unit_vectors:
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(max(min(dot_product, 1), -1))

            if angle > curvature_threshold and angle < 3.1:
                curvature_points.append(current_point)

        return np.array(curvature_points)

    def mark_curvature_points(self, points, radius=5, color=(255, 0, 0)):
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            marked_image = gray2rgb(self.image)
        else:
            marked_image = self.image.copy()

        for point in tqdm(points, desc="Marking points"):
            rr, cc = circle_perimeter(point[0], point[1], radius=radius, shape=marked_image.shape[:2])
            marked_image[rr, cc] = color
            
        return marked_image
    
    def plot_unit_vectors(self, unit_vectors, vector_scale=5, arrow_width=0.005, arrow_color='red'):
        plt.figure(figsize=(10, 6))
        plt.imshow(self.image, cmap='gray')
        plt.title("Unit Vectors on Image")
        plt.axis('off')

        for (unit_vector_1, unit_vector_2, current_point) in unit_vectors:
            y, x = current_point  # Switching to image coordinates
            dx1, dy1 = unit_vector_1 * vector_scale
            dx2, dy2 = unit_vector_2 * vector_scale

            plt.arrow(x, y, dx1, dy1, head_width=arrow_width, head_length=arrow_width, fc=arrow_color, ec=arrow_color)
            plt.arrow(x, y, dx2, dy2, head_width=arrow_width, head_length=arrow_width, fc=arrow_color, ec=arrow_color)

        plt.show()