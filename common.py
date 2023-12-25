import os
import re
import matplotlib.pyplot as plt


def plot_image(image, title=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def compare_plots(image1, image2):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image1, cmap='gray')
    ax[0].set_title('1')
    ax[0].axis('off')

    ax[1].imshow(image2,  cmap='gray')
    ax[1].set_title('2')
    ax[1].axis('off')

    plt.show()


def compute_save_path(image_path, sub_name):
    directory, filename_with_extension = os.path.split(image_path)
    filename, extension = os.path.splitext(filename_with_extension)
    
    disease_match = re.search(r'(.+)_processed', directory)
    if disease_match:
        disease = disease_match.group(1)
    else:
        raise ValueError("Disease name could not be determined from the path.")

    new_filename = f"{filename}_{sub_name}{extension}"
    new_directory = directory.replace(f"{disease}_processed", f"{disease}_circled")
    save_path = os.path.join(new_directory, new_filename)

    os.makedirs(new_directory, exist_ok=True)
    return save_path