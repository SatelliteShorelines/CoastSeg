# Standard library imports
import colorsys
import logging
import os
from collections import ChainMap
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# coastsat imports
from coastsat import SDS_preprocess, SDS_tools
from matplotlib import colors as mcolors

# External dependencies imports
from matplotlib import gridspec
from matplotlib.axes import Axes

# Logger setup
logger = logging.getLogger(__name__)


def determine_layout(
    im_shape: Tuple[int, ...],
) -> Tuple[
    gridspec.GridSpec,
    Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    Dict[str, Any],
]:
    """
    Determines subplot layout based on image aspect ratio.

    Args:
        im_shape (Tuple[int, ...]): Image shape (height, width[, channels]).

    Returns:
        Tuple[gridspec.GridSpec, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], Dict[str, Any]]:
            GridSpec layout, axis indices, and legend settings.
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
    ax1: Axes,
    ax2: Axes,
    ax3: Axes,
    im_orig: np.ndarray,
    im_merged: np.ndarray,
    im_all: np.ndarray,
    sl_pix: np.ndarray,
    im_ref_buffer: Optional[np.ndarray],
    shoreline_extraction_area: List[np.ndarray],
    titles: List[str],
    class_mapping: Dict[int, str],
    legend_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plots three panels for visualizing shoreline extraction results.

    Args:
        ax1 (Axes): Axes for original image panel.
        ax2 (Axes): Axes for merged classes panel.
        ax3 (Axes): Axes for all classes panel.
        im_orig (np.ndarray): Original input image.
        im_merged (np.ndarray): Image showing merged classes.
        im_all (np.ndarray): Image showing all classification results.
        sl_pix (np.ndarray): Pixel coordinates of extracted shoreline.
        im_ref_buffer (Optional[np.ndarray]): Reference shoreline buffer mask.
        shoreline_extraction_area (List[np.ndarray]): Shoreline extraction areas.
        titles (List[str]): Panel titles.
        class_mapping (Dict[int, str]): Mapping of class indices to names.
        legend_settings (Optional[Dict[str, Any]], optional): Legend customization settings.
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
        mask = np.ma.masked_where(~im_ref_buffer, im_ref_buffer)
        ax2.imshow(mask, cmap="PiYG", alpha=0.40)
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
        mask = np.ma.masked_where(~im_ref_buffer, im_ref_buffer)
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
) -> None:
    """
    Plots optical shoreline detection with RGB composite, classified output, and cloud mask.

    Args:
        im_ms (np.ndarray): Multispectral image array (HxWxBands).
        all_labels (np.ndarray): 2D array of all class labels.
        merged_labels (np.ndarray): 2D array where 1 indicates water, 0 indicates land.
        cloud_mask (np.ndarray): Boolean array indicating cloud pixels.
        sl_pix (np.ndarray): Detected shoreline pixel coordinates.
        class_mapping (Dict[int, Tuple[float, float, float]]): Mapping of class values to names.
        date (str): Acquisition date string.
        satname (str): Satellite name.
        im_ref_buffer (Optional[np.ndarray]): Reference shoreline buffer mask.
        output_path (str): Directory to save output image.
        shoreline_extraction_area (List[np.ndarray]): Shoreline extraction area coordinates.
        titles (Optional[List[str]], optional): Panel titles.
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies cloud mask to three images by setting cloudy pixels to 1.0.

    Args:
        im_RGB (np.ndarray): RGB image with shape (height, width, 3).
        im_merged (np.ndarray): Merged image with same shape as im_RGB.
        im_all (np.ndarray): All classes image with same shape as im_RGB.
        cloud_mask (np.ndarray): Boolean cloud mask with shape (height, width).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Masked RGB, merged, and all images.
    """
    nan_color_float = 1.0
    new_cloud_mask = np.repeat(cloud_mask[:, :, np.newaxis], im_RGB.shape[2], axis=2)

    im_RGB[new_cloud_mask] = nan_color_float
    im_merged[new_cloud_mask] = nan_color_float
    im_all[new_cloud_mask] = nan_color_float

    return im_RGB, im_merged, im_all


def _normalize_grayscale(im: np.ndarray) -> np.ndarray:
    """
    Normalizes grayscale image to [0, 1] using percentile-based contrast stretching.

    Args:
        im (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Normalized image with values in range [0, 1].
    """
    im = np.nan_to_num(im, nan=0.0)
    im_min, im_max = np.percentile(im, [2, 98])
    return np.clip((im - im_min) / (im_max - im_min), 0, 1)


def _convert_color(color: Tuple) -> Tuple[float, float, float]:
    """
    Convert hex or int RGB to normalized RGB.

    Args:
        color (Tuple): Color as hex string or RGB tuple.

    Returns:
        Tuple[float, float, float]: Normalized RGB values [0, 1].
    """
    if isinstance(color, str):
        return mcolors.to_rgb(color)
    return np.array(color) / 255


def add_legend(
    ax: Axes,
    include_extraction_area: bool,
    class_mapping: Dict[int, str],
    color_mapping: Dict[int, Tuple[float, float, float]],
    legend_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Adds a legend to the axes for optical shoreline classification.

    Args:
        ax (Axes): The axis to add legend to.
        include_extraction_area (bool): Whether to include shoreline extraction area.
        class_mapping (Dict[int, str]): Mapping of class indices to names.
        color_mapping (Dict[int, Tuple[float, float, float]]): Mapping of class indices to colors.
        legend_settings (Optional[Dict[str, Any]], optional): Legend customization settings.
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
    class_mapping: Dict[int, str],
    save_location: str = "",
    cloud_mask: Optional[np.ndarray] = None,
    im_ref_buffer: Optional[np.ndarray] = None,
    shoreline_extraction_area: Optional[List[np.ndarray]] = None,
    is_sar: bool = False,
) -> bool:
    """
    Unified function for plotting shoreline detection on SAR and optical images.

    Args:
        im_ms (np.ndarray): Image array (SAR or multispectral).
        all_labels (np.ndarray): All class labels (2D array of integers).
        merged_labels (np.ndarray): Array where 1 indicates water, 0 indicates land.
        shoreline (np.ndarray): Shoreline coordinates (X, Y) in projected coordinates.
        image_epsg (int): EPSG code of image CRS.
        georef (np.ndarray): Georeferencing array from the source image.
        settings (Dict[str, Any]): Configuration dictionary.
        date (str): Date of image.
        satname (str): Satellite name.
        class_mapping (Dict[int, str]): Mapping of class values to names.
        save_location (str, optional): Path to store outputs.
        cloud_mask (Optional[np.ndarray], optional): Cloud pixel mask (optical only).
        im_ref_buffer (Optional[np.ndarray], optional): Reference shoreline buffer.
        shoreline_extraction_area (Optional[List[np.ndarray]], optional): Shoreline extraction area coordinates.
        is_sar (bool, optional): Whether input is SAR imagery.

    Returns:
        bool: True if user accepted detections, False otherwise.
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
    Create an overlay on image using labels with specified opacity.

    Args:
        im_RGB (np.ndarray): Input RGB image (height, width, 3).
        im_labels (np.ndarray): Integer labels array with same dimensions as image.
        overlay_opacity (float, optional): Overlay opacity value. Defaults to 0.35.

    Returns:
        np.ndarray: Combined image and overlay array.
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
    Transforms world coordinates to pixel coordinates using transformation function.

    Args:
        shoreline_extraction_area (Optional[List[np.ndarray]]): List of world coordinates or None.
        output_epsg (int): EPSG code for world coordinate system.
        georef (np.ndarray): Georeferencing metadata for transformation.
        image_epsg (int): EPSG code of image coordinate system.
        transform_func: Function to perform world-to-pixel transformation.

    Returns:
        List[np.ndarray]: Transformed pixel coordinates or empty list if input is None.
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
    class_mapping: Dict[int, str],
    date: str,
    satname: str,
    im_ref_buffer: Optional[np.ndarray],
    output_path: str,
    shoreline_extraction_area: List[np.ndarray],
    titles: Optional[List[str]] = None,
) -> bool:
    """
    Plots SAR detection results with grayscale image, shoreline pixels, and labeled areas.

    Args:
        im_ms (np.ndarray): Multispectral or grayscale image to display.
        all_labels (np.ndarray): 2D array of all class labels.
        merged_labels (np.ndarray): 2D array where 1 indicates water, 0 indicates land.
        sl_pix (np.ndarray): Shoreline pixel coordinates (Nx2).
        class_mapping (Dict[int, str]): Mapping of class values to names.
        date (str): Date string for display and file naming.
        satname (str): Satellite name for display and file naming.
        im_ref_buffer (Optional[np.ndarray]): Reference shoreline buffer mask.
        output_path (str): Directory path to save the figure.
        shoreline_extraction_area (List[np.ndarray]): Pixel coordinates of extraction area.
        titles (Optional[List[str]], optional): Panel titles.

    Returns:
        bool: Always returns False (placeholder return value).
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
    save_detection_figure(fig, output_path, date, satname)

    plt.close(fig)
    return False


def transform_shoreline(
    shoreline: np.ndarray, output_epsg: int, image_epsg: int, georef: np.ndarray
) -> np.ndarray:
    """
    Transforms shoreline coordinates between EPSG projections and converts to pixel coordinates.

    Args:
        shoreline (np.ndarray): Shoreline coordinates in original projection.
        output_epsg (int): EPSG code of target projection.
        image_epsg (int): EPSG code of image projection.
        georef (np.ndarray): Georeferencing information for pixel conversion.

    Returns:
        np.ndarray: Shoreline coordinates in pixel space, or NaN array if error occurs.

    """
    try:
        shoreline_proj = SDS_tools.convert_epsg(shoreline, output_epsg, image_epsg)
        return SDS_tools.convert_world2pix(shoreline_proj[:, :2], georef)
    except Exception:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])


def _prepare_output_dir(
    output_dir: Optional[str], base_path: str, sitename: str
) -> str:
    """
    Creates and ensures existence of output directory for detection images.

    Args:
        output_dir (Optional[str]): Optional base directory for output.
        base_path (str): Base path to use if output_dir is not provided.
        sitename (str): Site name for directory path construction.

    Returns:
        str: Full path to the created output directory.
    """
    path = os.path.join(
        output_dir or os.path.join(base_path, sitename), "jpg_files", "detection"
    )
    os.makedirs(path, exist_ok=True)
    return path


def create_color_mapping_as_ints(
    int_list: List[int],
) -> Dict[int, Tuple[int, int, int]]:
    """
    Creates color mapping dictionary with RGB values as integers (0-255).

    Args:
        int_list (List[int]): List of integers for color generation.

    Returns:
        Dict[int, Tuple[int, int, int]]: Dictionary mapping integers to RGB color tuples.
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(h, 0.5, 1.0)]
        color_mapping[num] = (r, g, b)

    return color_mapping


def create_color_mapping(int_list: List[int]) -> Dict[int, Tuple[float, float, float]]:
    """
    Creates color mapping dictionary with RGB values as floats (0.0-1.0).

    Args:
        int_list (List[int]): List of integers for color generation.

    Returns:
        Dict[int, Tuple[float, float, float]]: Dictionary mapping integers to RGB color tuples.
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [x for x in colorsys.hls_to_rgb(h, 0.5, 1.0)]
        color_mapping[num] = (r, g, b)

    return color_mapping


def create_classes_overlay_image(labels: np.ndarray) -> np.ndarray:
    """
    Creates overlay image by mapping class labels to colors.

    Args:
        labels (np.ndarray): 2D array of integer class labels for each pixel.

    Returns:
        np.ndarray: 3D array representing overlay image with same size as input labels.
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


def save_detection_figure(
    fig: plt.Figure, filepath: str, date: str, satname: str
) -> None:
    """
    Save figure as JPG file with specified DPI.

    Args:
        fig (plt.Figure): Figure object to save.
        filepath (str): Directory path for saving image.
        date (str): Date in format 'YYYYMMDD'.
        satname (str): Satellite name.
    """
    fig.savefig(
        os.path.join(filepath, date + "_" + satname + ".jpg"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure after saving
    plt.close("all")
    del fig
