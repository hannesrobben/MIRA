import matplotlib.pyplot as plt


def plot_structure_properties(
    structure, reference_structure=None, suptitle=None
) -> None:
    # shortcut for plotting
    if suptitle is None:
        suptitle = "Image Processing Steps"

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(suptitle, fontsize=16)
    axs[0, 0].imshow(structure.data, cmap="gray")
    axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(structure.mask, cmap="gray")
    axs[0, 1].set_title("structure Image")
    # axs[1, 0].imshow(structure.corner_points, cmap="gray")
    # axs[1, 0].set_title("edges Image")
    # axs[1, 1].imshow(structure.bounds, cmap="gray")
    # axs[1, 1].set_title("Bound Image")
    plt.show()
    if reference_structure is not None:
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle(suptitle, fontsize=16)
        axs[0, 0].imshow(reference_structure.data, cmap="gray")
        axs[0, 0].set_title("Original Image")
        axs[0, 1].imshow(reference_structure.structure_mask, cmap="gray")
        axs[0, 1].set_title("structure Image")
        axs[1, 0].imshow(reference_structure._edges, cmap="gray")
        axs[1, 0].set_title("edges Image")
        axs[1, 1].imshow(reference_structure._edge_continuous, cmap="gray")
        axs[1, 1].set_title("edges continuous Image")
        plt.show()
