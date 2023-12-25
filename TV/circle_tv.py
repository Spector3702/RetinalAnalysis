import numpy as np
from skimage import io, morphology, filters

from tv_processer import TVProcesser
from common import compute_save_path


def double_skeletonize(vessels_image):
    binarized_image = vessels_image < np.mean(vessels_image)
    skeleton = morphology.skeletonize(binarized_image)

    blurred_skeleton = filters.gaussian(skeleton, sigma=5)
    re_threshold_value = filters.threshold_otsu(blurred_skeleton)
    re_binarized_skeleton = blurred_skeleton > re_threshold_value
    re_skeleton = morphology.skeletonize(re_binarized_skeleton)

    return re_skeleton


def circle_tv(image_path):
    vessels_image = io.imread(image_path, as_gray=True)
    re_skeleton = double_skeletonize(vessels_image)

    tv_processer = TVProcesser(vessels_image)
    unit_vectors = tv_processer.get_unit_vectors(re_skeleton, 5)
    curvature_points = tv_processer.calculate_curvature_from_vectors(unit_vectors, curvature_threshold=np.pi/4)

    marked_acute_image = tv_processer.mark_curvature_points(curvature_points, radius=5, color=(255, 0, 0))
    save_path = compute_save_path(image_path, 'circled')
    io.imsave(save_path, marked_acute_image)


if __name__ == '__main__':
    image_path = 'data/Sample_imgs/TV_processed/907_bloodvessel.png'
    circle_tv(image_path)