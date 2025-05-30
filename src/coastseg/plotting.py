# Standard library imports
import colorsys
import os
import logging
from typing import Any, Dict, Optional, Tuple, List
from collections import ChainMap

# External dependencies imports
from matplotlib import gridspec
import matplotlib.lines as mlines
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# coastsat imports
from coastsat import SDS_preprocess, SDS_tools

# Logger setup
logger = logging.getLogger(__name__)


def determine_layout(im_shape):
    """
    Determines the appropriate subplot layout based on image aspect ratio. By default, it uses a horizontal layout.

    Args:
        im_shape (tuple): Shape of the image (height, width[, channels])

    Returns:
        gs (GridSpec): GridSpec layout
        ax_indices (tuple): ((ax1_idx), (ax2_idx), (ax3_idx)) as grid positions
        legend_settings (dict): Dict with bbox_to_anchor and loc for legends
    """
    height, width = im_shape[:2]
    if height > 2.5 * width:  # This means the image is vertical
        gs = gridspec.GridSpec(3, 1)
        ax_indices = ((0, 0), (1, 0), (2, 0))
        legend_settings = {"bbox_to_anchor": (1.05, 0.5), "loc": "center left"}
    else:  # This means the image is horizontal (default)
        gs = gridspec.GridSpec(1, 3)
        ax_indices = ((0, 0), (0, 1), (0, 2))
        legend_settings = {"bbox_to_anchor": (0.5, -0.23), "loc": "lower center"}

    return gs, ax_indices, legend_settings


def plot_shared_panels(
    ax1,
    ax2,
    ax3,
    im_orig,
    im_merged,
    im_all,
    sl_pix,
    im_ref_buffer,
    shoreline_extraction_area,
    titles,
    class_mapping,
    legend_settings: Optional[Dict[str, Any]] = None,
):
    """
    Plots a set of three panels for visualizing shoreline extraction results.

    Parameters:
    -----------
    ax1, ax2, ax3 : matplotlib.axes.Axes
        Axes objects where the original image, merged classes, and all classes will be plotted respectively.

    im_orig : np.ndarray
        The original input image (e.g., satellite image) to be displayed in the first panel.

    im_merged : np.ndarray
        The image showing merged classes (e.g., binary classification of water and land), displayed in the second panel.

    im_all : np.ndarray
        The image showing all classification results (multi-class), displayed in the third panel.

    sl_pix : np.ndarray
        Pixel coordinates of the extracted shoreline to be plotted on all panels (shape: [N, 2]).

    im_ref_buffer : np.ndarray or None
        Boolean mask indicating reference shoreline buffer to be overlaid on the second and third panels. If None, no buffer is shown.

    shoreline_extraction_area : List[np.ndarray]
        List of arrays representing different shoreline extraction areas, each array of shape [M, 2].

    titles : List[str]
        Titles for the three panels in the order: [original image, merged classes, all classes].

    class_mapping : Dict[int, str]
        Dictionary mapping class indices to human-readable class names. Used in the legend for the third panel.

    legend_settings : Optional[Dict[str, Any]], default=None
        Additional settings to customize the legend (e.g., location, font size).

    Returns:
    --------
    None
        The function modifies the input axes in-place with the respective plots.
    """
    # the First plot the original image
    ax1.imshow(np.nan_to_num(im_orig))
    ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for idx in range(len(shoreline_extraction_area)):
        ax1.plot(
            shoreline_extraction_area[idx][:, 0],
            shoreline_extraction_area[idx][:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax1.set_title(titles[0])
    ax1.axis("off")

    # the second plot the merged classes
    ax2.imshow(im_merged)  # plot the land and water classes
    # plot the reference shoreline buffer on top of the merged classes
    if im_ref_buffer is not None:
        mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
        ax2.imshow(mask, cmap="PiYG", alpha=0.30)
    ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for idx in range(len(shoreline_extraction_area)):
        ax2.plot(
            shoreline_extraction_area[idx][:, 0],
            shoreline_extraction_area[idx][:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax2.set_title(titles[1])
    ax2.axis("off")

    # for the merged classes 0 values represent land/other and 1 values represent water
    add_legend(
        ax2,
        bool(shoreline_extraction_area),
        class_mapping={0: "other", 1: "water"},
        color_mapping={},
        legend_settings=legend_settings,
    )

    # the third panel plots the all classes
    ax3.imshow(im_all)
    # plot the reference shoreline buffer
    if im_ref_buffer is not None:
        mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
        ax3.imshow(mask, cmap="PiYG", alpha=0.30)
    ax3.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for idx in range(len(shoreline_extraction_area)):
        ax3.plot(
            shoreline_extraction_area[idx][:, 0],
            shoreline_extraction_area[idx][:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax3.set_title(titles[2])
    ax3.axis("off")

    # add a legend to the third panel that shows the classes
    add_legend(
        ax3,
        bool(shoreline_extraction_area),
        class_mapping,
        color_mapping={},
        legend_settings=legend_settings,
    )


def plot_optical_detection(
    im_ms: np.ndarray,
    all_labels: np.ndarray,
    merged_labels: np.ndarray,
    cloud_mask: np.ndarray,
    sl_pix: np.ndarray,
    class_mapping: Dict[int, Tuple[float, float, float]],
    date: str,
    satname: str,
    im_ref_buffer: Optional[np.ndarray],
    output_path: str,
    shoreline_extraction_area: List[np.ndarray],
    titles: Optional[List[str]] = None,
):
    """
    Plots optical shoreline detection including RGB composite, classified output,
    NDWI index, cloud mask, and reference buffer.

    Args:
        im_ms (np.ndarray): Multispectral image array (HxWxBands).
        all_labels (np.ndarray): 2D array of all class labels as integers.Eg [[0, 1, 2, 3],[0,2,2,3]]
        merged_labels (np.ndarray): 2D array of 0 and 1 where 1 indicates water and 0 indicates land/other
        cloud_mask (np.ndarray): Boolean array indicating cloud-covered pixels.
        sl_pix (np.ndarray): Nx2 array of detected shoreline pixel coordinates.
        class_mapping (Dict[int, str): Mapping of class values to the class name.
            example: {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
        date (str): Acquisition date string.
            example: '20231001' for October 1, 2023.
        satname (str): Name of the satellite (for display and filename).
            example:"L8" for Landsat 8.
        im_ref_buffer (Optional[np.ndarray]): Boolean mask for reference shoreline buffer.
        output_path (str): Directory to save the output image.
        shoreline_extraction_area (list[np.ndarry]): The shoreline extraction area in pixel coordinates.
        sitename (str): Name of the site (for display in title).

    """
    if not titles or len(titles) != 3:
        titles = ["Original Image", "Merged Classes", "All Classes"]

    # Normalize the multispectral image for display
    im_rgb = SDS_preprocess.rescale_image_intensity(
        im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9
    )

    # Layout based on shape
    gs, (idx1, idx2, idx3), legend_settings = determine_layout(im_rgb.shape)
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(gs[idx1])
    ax2 = fig.add_subplot(gs[idx2])
    ax3 = fig.add_subplot(gs[idx3])

    im_merged = create_overlay(im_rgb, merged_labels, overlay_opacity=0.35)
    im_all = create_overlay(im_rgb, all_labels, overlay_opacity=0.35)

    # Mask clouds in the images
    im_rgb, im_merged, im_all = mask_clouds_in_images(
        im_rgb, im_merged, im_all, cloud_mask
    )

    plot_shared_panels(
        ax1,
        ax2,
        ax3,
        im_rgb,
        im_merged,
        im_all,
        sl_pix,
        im_ref_buffer,
        shoreline_extraction_area,
        titles,
        class_mapping,
        legend_settings=legend_settings,
    )

    # save a .jpg under /jpg_files/detection
    save_detection_figure(fig, output_path, date, satname)
    plt.close(fig)


def mask_clouds_in_images(
    im_RGB: np.ndarray,
    im_merged: np.ndarray,
    im_all: np.ndarray,
    cloud_mask: np.ndarray,
):
    """
    Applies a cloud mask to three input images (im_RGB, im_merged & im_all) by setting the
    cloudy portions to a value of 1.0.

    Args:
        im_RGB (np.ndarray[float]): An RGB image, with shape (height, width, 3).
        im_merged (np.ndarray[float]): A merged image, with the same shape as im_RGB.
        im_all (np.ndarray[float]): An 'all' image, with the same shape as im_RGB.
        cloud_mask (np.ndarray[bool]): A boolean cloud mask, with shape (height, width).

    Returns:
        tuple: A tuple containing the masked im_RGB, im_merged and im_all images.
    """
    nan_color_float = 1.0
    new_cloud_mask = np.repeat(cloud_mask[:, :, np.newaxis], im_RGB.shape[2], axis=2)

    im_RGB[new_cloud_mask] = nan_color_float
    im_merged[new_cloud_mask] = nan_color_float
    im_all[new_cloud_mask] = nan_color_float

    return im_RGB, im_merged, im_all


def _normalize_grayscale(im):
    """
    Normalizes a grayscale image array to the range [0, 1] using percentile-based contrast stretching.

    This function replaces NaN values in the input image with 0.0, then computes the 2nd and 98th percentiles
    to determine the lower and upper bounds for normalization. The image is then scaled so that values below
    the 2nd percentile become 0, values above the 98th percentile become 1, and all other values are linearly
    scaled between 0 and 1.

    Parameters:
        im (np.ndarray): Input grayscale image as a NumPy array.

    Returns:
        np.ndarray: Normalized image array with values in the range [0, 1].
    """
    im = np.nan_to_num(im, nan=0.0)
    im_min, im_max = np.percentile(im, [2, 98])
    return np.clip((im - im_min) / (im_max - im_min), 0, 1)


def _convert_color(color: Tuple):
    """Convert hex or int RGB to normalized RGB."""
    if isinstance(color, str):
        return mcolors.to_rgb(color)
    return np.array(color) / 255


def add_legend(
    ax: Axes,
    include_extraction_area: bool,
    class_mapping: dict,
    color_mapping: dict,
    legend_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Adds a legend to the provided Axes object for optical shoreline classification.

    ax (matplotlib.axes.Axes): The axis to which the legend will be added.
    include_extraction_area (bool): Whether to include the shoreline extraction area in the legend.
    class_mapping (dict): A dictionary mapping class indices to class names.
    color_mapping (dict, optional): A dictionary mapping class indices to colors. Defaults to None.
        If None, a default color mapping will be created.

    Returns:
        None: The function modifies the Axes object in place.

    """
    if not legend_settings:
        legend_settings = {}

    defaults = {
        "bbox_to_anchor": (0.5, -0.23),
        "loc": "lower center",
        "borderaxespad": 0.0,
        "fontsize": 10,
    }
    legend_settings = dict(ChainMap(legend_settings, defaults))

    if not color_mapping:
        color_mapping = create_color_mapping_as_ints(list(class_mapping.keys()))

    patches_for_classes = [
        mpatches.Patch(
            color=tuple(_convert_color(color)),
            label=f"{class_mapping.get(index, f'{index}')}",
        )
        for index, color in color_mapping.items()
    ]

    handles = [
        mlines.Line2D([], [], color="k", label="shoreline"),
        mpatches.Patch(color="#800000", alpha=0.8, label="reference shoreline buffer"),
    ]

    if include_extraction_area:
        handles.append(
            mlines.Line2D([], [], color="#cb42f5", label="shoreline extraction area")
        )

    ax.legend(handles=handles + patches_for_classes, **legend_settings)


def plot_detection(
    im_ms: np.ndarray,
    all_labels: np.ndarray,
    merged_labels: np.ndarray,
    shoreline: np.ndarray,
    image_epsg: int,
    georef: np.ndarray,
    settings: Dict[str, Any],
    date: str,
    satname: str,
    class_mapping: dict,
    save_location: str = "",
    cloud_mask: Optional[np.ndarray] = None,
    im_ref_buffer: Optional[np.ndarray] = None,
    shoreline_extraction_area: Optional[List[np.ndarray]] = None,
    is_sar: bool = False,
):
    """
    Unified function for plotting shoreline detection on SAR and optical images.
    Saves the plot and optionally prompts user input to skip/accept detections.

    Parameters:
    - im_ms: np.array - image array (SAR or multispectral)
    - all_labels : np.array - all class labels (2D array of integers)
    - merged_labels: np.array - an array of 0 and 1 where 1 indicates water and 0 indicates land/other
    - shoreline: np.array - coordinates of shoreline (X, Y) (eg. projected coordinates, NOT pixel coordinates)
    - image_epsg: int - EPSG code of image CRS that the shoreline was derived from
    - georef: np.array - georeferencing array from the tif that the shoreline was derived from
    - settings: dict - config dictionary
    - date: str - date of image
    - satname: str - satellite name
    - class_mapping (Dict[int, str): Mapping of class values to the class name.
        example: {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    - cloud_mask: np.array - mask of cloud pixels (optical only)
    - im_ref_buffer: np.array - reference shoreline buffer
    - output_directory: str - path to store outputs
    - shoreline_extraction_area (Optional[List[np.ndarray]]): the shoreline extraction area in geographic coordinates
        This is a list of np.ndarrays, where each ndarray is a 2D array of coordinates (X, Y) in the geographic coordinate system.
    - is_sar: bool - whether the input is SAR imagery

    Returns
        bool: True if the user accepted the detections, False otherwise.

    """
    sitename = settings["inputs"]["sitename"]
    filepath_data = save_location if save_location else settings["inputs"]["filepath"]
    # this is the 'filepath' where the images will be saved aka session_name/sitename/jpg_files/detection
    output_path = _prepare_output_dir(save_location, filepath_data, sitename)
    # turn the shoreline into pixel coordinates
    sl_pix = transform_shoreline(shoreline, settings["output_epsg"], image_epsg, georef)

    shoreline_detection_area_pix = transform_shoreline_area_to_pixel_coords(
        shoreline_extraction_area,
        output_epsg=settings["output_epsg"],
        georef=georef,
        image_epsg=image_epsg,
        transform_func=SDS_preprocess.transform_world_coords_to_pixel_coords,
    )

    if is_sar:
        return plot_sar_detection(
            im_ms,
            all_labels,
            merged_labels,
            sl_pix,
            class_mapping,
            date,
            satname,
            im_ref_buffer,
            output_path,
            shoreline_detection_area_pix,
            titles=["Original Image", "Merged Classes", "All Classes"],
        )
    else:
        # if cloud_mask is None, create a default mask of zeros
        cloud_mask = (
            cloud_mask
            if cloud_mask is not None
            else np.zeros_like(im_ms[:, :, 0], dtype=bool)
        )
        return plot_optical_detection(
            im_ms,
            all_labels,
            merged_labels,
            cloud_mask,
            sl_pix,
            class_mapping,
            date,
            satname,
            im_ref_buffer,
            output_path,
            shoreline_detection_area_pix,
            titles=["Original Image", "Merged Classes", "All Classes"],
        )


def create_overlay(
    im_RGB: np.ndarray,
    im_labels: np.ndarray,
    overlay_opacity: float = 0.35,
) -> np.ndarray:
    """
    Create an overlay on the given image using the provided labels and
    specified overlay opacity.

    Args:
    im_RGB (np.ndarray[float]): The input image as an RGB numpy array (height, width, 3).
    im_labels (np.ndarray[int]): The array containing integer labels of the same dimensions as the input image.
    overlay_opacity (float, optional): The opacity value for the overlay (default: 0.35).

    Returns:
    np.ndarray[float]: The combined numpy array of the input image and the overlay.
    """
    # Create an overlay using the given labels
    overlay = create_classes_overlay_image(im_labels)
    # Combine the original image and the overlay using the correct opacity
    combined_float = im_RGB * (1 - overlay_opacity) + overlay * overlay_opacity
    return combined_float


def transform_shoreline_area_to_pixel_coords(
    shoreline_extraction_area: Optional[List[np.ndarray]],
    output_epsg: int,
    georef: np.ndarray,
    image_epsg: int,
    transform_func,
) -> List[np.ndarray]:
    """
    Transforms a list of world coordinates into pixel coordinates using a transformation function.

    Args:
        shoreline_extraction_area (Optional[List[np.ndarray]]):
            The input list of world coordinates (or None).
            Example:
            [array([[ 606270.9268185 , 4079016.66886246],
                    [ 606967.96694302, 4077461.14324242],
                    [ 607305.27201052, 4076390.54086919],
                    [ 607289.08625894, 4076373.12052936],
                    [ 606690.57810944, 4076373.12052936],
                    [ 606061.90757097, 4077987.18491909],
                    [ 605284.49195324, 4079756.58371796],
                    [ 604687.47197666, 4080853.12052936],
                    [ 605266.43466503, 4080853.12052936],
                    [ 606270.9268185 , 4079016.66886246]])]
        output_epsg (int): EPSG code for the world coordinate system.
        georef (np.ndarray): Georeferencing metadata used for transformation.
            Example:  [ 6.04075722e+05  1.00000000e+01  0.00000000e+00  4.08085444e+06, 0.00000000e+00 -1.00000000e+01]
        image_epsg (int): EPSG code of the image coordinate system.
            Example : 32610 for UTM zone 10N.
        transform_func (callable): Function to perform the world-to-pixel transformation.
            Example: SDS_preprocess.transform_world_coords_to_pixel_coords - default transformation function.

    Returns:
        the transformed pixel coordinates as a list of tuples (x, y) or an empty list if the input is None or empty.
        Example:
        Shoreline extraction area in pixel coordinates:
            [array([[ 2.20236444e+02,  1.83645167e+02],
                [ 2.89940456e+02,  3.39197729e+02],
                [ 3.23670963e+02,  4.46257966e+02],
                [ 3.22052388e+02,  4.48000000e+02],
                [ 2.62201573e+02,  4.48000000e+02],
                [ 1.99334519e+02,  2.86593561e+02],
                [ 1.21592957e+02,  1.09653681e+02],
                [ 6.18909597e+01, -5.82076609e-11],
                [ 1.19787229e+02, -5.82076609e-11],
                [ 2.20236444e+02,  1.83645167e+02]])]
    """
    if not shoreline_extraction_area:
        return []

    if shoreline_extraction_area is not None:
        if len(shoreline_extraction_area) == 0:
            return []

    pixel_coords = [
        transform_func(coord, output_epsg, georef, image_epsg)
        for coord in shoreline_extraction_area
    ]

    return pixel_coords


def plot_sar_detection(
    im_ms: np.ndarray,
    all_labels: np.ndarray,
    merged_labels: np.ndarray,
    sl_pix: np.ndarray,
    class_mapping: dict,
    date: str,
    satname: str,
    im_ref_buffer: Optional[np.ndarray],
    output_path: str,
    shoreline_extraction_area: List[np.ndarray],
    titles: Optional[List[str]] = None,
) -> bool:
    """
    Plots SAR detection results including grayscale image, shoreline pixels,
    labeled water/land areas, and an optional reference shoreline buffer.

    Args:
        im_ms (np.ndarray): Multispectral or grayscale image to display.
        all_labels (np.ndarray): 2D array of all class labels as integers.Eg [[0, 1, 2, 3],[0,2,2,3]]
        merged_labels (np.ndarray): 2D array of 0 and 1 where 1 indicates water and 0 indicates land/other
        sl_pix (np.ndarray): Array of shoreline pixel coordinates (Nx2).
        class_mapping (Dict[int, str): Mapping of class values to the class name.
            example: {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
        date (str): Date string for display and file naming.
            example: '20231001' for October 1, 2023.
        satname (str): Satellite name for display and file naming.
            example:"L8" for Landsat 8.
        im_ref_buffer (Optional[np.ndarray]): Boolean mask of the reference shoreline buffer.
        output_path (str): Directory path to save the figure if enabled.
        settings (Dict[str, Any]): Dictionary of settings. Must contain key 'save_figure'.
        shoreline_extraction_area ([np.ndarray]): Pixel coordinates of the shoreline extraction area.
    Returns:
        bool: Always returns False. Used as a placeholder return value.
    """

    if not titles or len(titles) != 3:
        titles = ["Original Image", "Merged Classes", "All Classes"]

    # Normalize the multispectral image for display
    im_display = _normalize_grayscale(im_ms)

    # Layout based on shape
    gs, (idx1, idx2, idx3), legend_settings = determine_layout(im_display.shape)
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(gs[idx1])
    ax2 = fig.add_subplot(gs[idx2])
    ax3 = fig.add_subplot(gs[idx3])

    # Create overlays for merged and all labels
    im_merged = create_overlay(im_display, merged_labels, overlay_opacity=0.35)
    im_all = create_overlay(im_display, all_labels, overlay_opacity=0.35)

    plot_shared_panels(
        ax1,
        ax2,
        ax3,
        im_display,
        im_merged,
        im_all,
        sl_pix,
        im_ref_buffer,
        shoreline_extraction_area,
        titles,
        class_mapping,
        legend_settings=legend_settings,
    )

    # # Left panel
    # ax1.imshow(im_display, cmap="gray")
    # if im_ref_buffer is not None:
    #     mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
    #     ax1.imshow(mask, cmap="PiYG", alpha=0.6)
    # ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    # for idx in range(len(shoreline_extraction_area)):
    #     ax1.plot(
    #         shoreline_extraction_area[idx][:, 0],
    #         shoreline_extraction_area[idx][:, 1],
    #         color="#cb42f5",
    #         markersize=1,
    #     )
    # ax1.set_title(titles[0])
    # ax1.axis("off")

    # # the second plot the merged classes
    # ax2.imshow(im_merged) # plot the land and water classes
    # if im_ref_buffer is not None:
    #     ax2.imshow(
    #         np.ma.masked_where(~im_ref_buffer, im_ref_buffer), cmap="PiYG", alpha=0.5
    #     )
    # ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    # for idx in range(len(shoreline_extraction_area)):
    #     ax2.plot(
    #         shoreline_extraction_area[idx][:, 0],
    #         shoreline_extraction_area[idx][:, 1],
    #         color="#cb42f5",
    #         markersize=1,
    #     )

    # add_legend(ax2,bool(shoreline_extraction_area),class_mapping={0: "other", 1: "water"},color_mapping={})
    # ax2.set_title(titles[1])
    # ax2.axis("off")

    # # the third panel plots the all classes
    # ax3.imshow(im_all)
    # # plot the reference shoreline buffer
    # if im_ref_buffer is not None:
    #     mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
    #     ax3.imshow(mask, cmap="PiYG", alpha=0.30)
    # ax3.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    # for idx in range(len(shoreline_extraction_area)):
    #     ax3.plot(
    #         shoreline_extraction_area[idx][:, 0],
    #         shoreline_extraction_area[idx][:, 1],
    #         color="#cb42f5",
    #         markersize=1,
    #     )
    # # add a legend to the third panel that shows the classes
    # add_legend(ax3, bool(shoreline_extraction_area),class_mapping,color_mapping={})

    # ax3.set_title(titles[2] )
    # ax3.axis("off")
    # save a .jpg under /jpg_files/detection
    save_detection_figure(fig, output_path, date, satname)

    plt.close(fig)
    return False


def transform_shoreline(
    shoreline: np.ndarray, output_epsg: int, image_epsg: int, georef: np.ndarray
) -> np.ndarray:
    """
    Transforms a shoreline's coordinates from one EPSG projection to another and converts them to pixel coordinates.

    Args:
        shoreline (np.ndarray): Array of shoreline coordinates in the original projection.
            Example: [[606270.9268185, 4079016.66886246], [606967.96694302, 4077461.14324242], ...]
        output_epsg (int): EPSG code of the target projection.
        image_epsg (int): EPSG code of the image's projection.
        georef (np.ndarray): Georeferencing information for converting world coordinates to pixel coordinates.

    Returns:
        np.ndarray: Array of shoreline coordinates in pixel space. If an error occurs, returns an array of NaN values.
        Example: [[120.   12.5],[120.5  13. ],[121.   14.5],[121.5  15. ] ... ]
            where each row represents a pixel coordinate (x, y) in the image.
    """
    try:
        shoreline_proj = SDS_tools.convert_epsg(shoreline, output_epsg, image_epsg)
        return SDS_tools.convert_world2pix(shoreline_proj[:, :2], georef)
    except Exception:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])


def _prepare_output_dir(output_dir, base_path, sitename):
    """
    Creates and ensures the existence of an output directory for detection images.

    This function constructs a directory path for storing detection images in JPEG format.
    If `output_dir` is provided, it is used as the base directory; otherwise, the path is
    constructed using `base_path` and `sitename`. The resulting path will have the structure:
    <output_dir or base_path/sitename>/jpg_files/detection. The directory is created if it
    does not already exist.

    Args:
        output_dir (str or None): Optional base directory for output. If None, uses base_path and sitename.
        base_path (str): The base path to use if output_dir is not provided.
        sitename (str): The site name to use in the directory path.

    Returns:
        str: The full path to the created output directory.
    """
    path = os.path.join(
        output_dir or os.path.join(base_path, sitename), "jpg_files", "detection"
    )
    os.makedirs(path, exist_ok=True)
    return path


def create_color_mapping_as_ints(int_list: list[int]) -> dict:
    """
    This function creates a color mapping dictionary for a given list of integers, assigning a unique RGB color to each integer. The colors are generated using the HLS color model, and the resulting RGB values are integers in the range of 0-255.

    Arguments:

    int_list (list): A list of integers for which unique colors need to be generated.
    Returns:

    color_mapping (dict): A dictionary where the keys are the input integers and the values are the corresponding RGB colors as tuples of integers.
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(h, 0.5, 1.0)]
        color_mapping[num] = (r, g, b)

    return color_mapping


def create_color_mapping(int_list: list[int]) -> dict:
    """
    This function creates a color mapping dictionary for a given list of integers, assigning a unique RGB color to each integer. The colors are generated using the HLS color model, and the resulting RGB values are floating-point numbers in the range of 0.0-1.0.

    Arguments:

    int_list (list): A list of integers for which unique colors need to be generated.
    Returns:

    color_mapping (dict): A dictionary where the keys are the input integers and the values are the corresponding RGB colors as tuples of floating-point numbers.
        Example : {0: (1.0, 0.0, 0.0), 1: (0.5000000000000002, 1.0, 0.0), 2: (0.0, 0.9999999999999998, 1.0), 3: (0.49999999999999956, 0.0, 1.0)}
        Example : {False: (1.0, 0.0, 0.0), True: (0.0, 0.9999999999999998, 1.0)}
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [x for x in colorsys.hls_to_rgb(h, 0.5, 1.0)]
        color_mapping[num] = (r, g, b)

    return color_mapping


def create_classes_overlay_image(labels):
    """
    Creates an overlay image by mapping class labels to colors.

    Args:
    labels (numpy.ndarray): A 2D array representing class labels for each pixel in an image.
    This is an np.ndarray of integer class labels, where each unique integer represents a different class.

    Returns:
    numpy.ndarray: A 3D array representing an overlay image with the same size as the input labels.
    """
    # Ensure that the input labels is a NumPy array
    labels = np.asarray(labels)

    # Make an overlay the same size of the image with 3 color channels
    overlay_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.float32)

    # Create a color mapping for the labels
    class_values = np.unique(labels)  # get only the unique classes
    color_mapping = create_color_mapping(list(class_values))

    # Create the overlay image by assigning the color for each label
    for class_value, class_color in color_mapping.items():
        overlay_image[labels == class_value] = class_color

    return overlay_image


def save_detection_figure(fig, filepath: str, date: str, satname: str) -> None:
    """
    Save the given figure as a jpg file with a specified dpi.

    Args:
    fig (Figure): The figure object to save.
    filepath (str): The directory path where the image will be saved.
    date (str): The date the satellite image was taken in the format 'YYYYMMDD'.
    satname (str): The name of the satellite that took the image.

    Returns:
    None
    """
    fig.savefig(
        os.path.join(filepath, date + "_" + satname + ".jpg"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure after saving
    plt.close("all")
    del fig
