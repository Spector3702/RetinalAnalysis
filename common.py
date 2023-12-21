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

    ax[1].imshow(image2)
    ax[1].set_title('2')
    ax[1].axis('off')

    plt.show()