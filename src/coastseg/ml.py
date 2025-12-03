import os
import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import scipy.ndimage

import tensorflow as tf  # numerical operations on gpu
import tensorflow.keras.backend as K

# This code was originally written by Dr. Daniel Buscombe and has been modified to work with the
# coastseg package. The original code can be found at https://github.com/Doodleverse/doodleverse_utils/blob/main/doodleverse_utils/prediction_imports.py


def label_to_colors(
    img,
    mask,
    alpha,  # =128,
    colormap,  # =class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,  # =0,
    do_alpha,  # =True
):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].

    """

    def fromhex(n):
        """hexadecimal to integer"""
        return int(n, base=16)

    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    cimg[mask == 1] = (0, 0, 0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg


def standardize(img):
    # standardization using adjusted standard deviation

    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0 / np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img) == 2:
        img = np.dstack((img, img, img))

    return img


def est_label_binary(image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE, w, h):
    E0 = []
    E1 = []

    for counter, model in enumerate(M):
        # heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)
        try:
            if MODEL == "segformer":
                # est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).logits
                est_label = model(tf.expand_dims(image, 0)).logits
            else:
                est_label = tf.squeeze(
                    model.predict(tf.expand_dims(image, 0), batch_size=1)
                )

        except Exception:
            if MODEL == "segformer":
                est_label = model.predict(
                    tf.expand_dims(image[:, :, 0], 0), batch_size=1
                ).logits
            else:
                est_label = tf.squeeze(
                    model.predict(tf.expand_dims(image[:, :, 0], 0), batch_size=1)
                )

        if TESTTIMEAUG:
            # return the flipped prediction
            if MODEL == "segformer":
                est_label2 = np.flipud(
                    model.predict(
                        tf.expand_dims(np.flipud(image), 0), batch_size=1
                    ).logits
                )
            else:
                est_label2 = np.flipud(
                    tf.squeeze(
                        model.predict(tf.expand_dims(np.flipud(image), 0), batch_size=1)
                    )
                )

            if MODEL == "segformer":
                est_label3 = np.fliplr(
                    model.predict(
                        tf.expand_dims(np.fliplr(image), 0), batch_size=1
                    ).logits
                )
            else:
                est_label3 = np.fliplr(
                    tf.squeeze(
                        model.predict(tf.expand_dims(np.fliplr(image), 0), batch_size=1)
                    )
                )

            if MODEL == "segformer":
                est_label4 = np.flipud(
                    np.fliplr(
                        model.predict(
                            tf.expand_dims(np.flipud(np.fliplr(image)), 0), batch_size=1
                        ).logits
                    )
                )
            else:
                est_label4 = np.flipud(
                    np.fliplr(
                        tf.squeeze(
                            model.predict(
                                tf.expand_dims(np.flipud(np.fliplr(image)), 0),
                                batch_size=1,
                            )
                        )
                    )
                )

            # soft voting - sum the softmax scores to return the new TTA estimated softmax scores
            est_label = est_label + est_label2 + est_label3 + est_label4
            # del est_label2, est_label3, est_label4

        est_label = est_label.numpy().astype("float32")

        if MODEL == "segformer":
            est_label = resize(
                est_label,
                (1, NCLASSES, TARGET_SIZE[0], TARGET_SIZE[1]),
                preserve_range=True,
                clip=True,
            ).squeeze()
            est_label = np.transpose(est_label, (1, 2, 0))

        E0.append(resize(est_label[:, :, 0], (w, h), preserve_range=True, clip=True))
        E1.append(resize(est_label[:, :, 1], (w, h), preserve_range=True, clip=True))
        # del est_label
    # heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)
    K.clear_session()

    return E0, E1


def est_label_multiclass(image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE):
    est_label = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], NCLASSES))

    for counter, model in enumerate(M):
        # heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)
        try:
            if MODEL == "segformer":
                est_label = model(tf.expand_dims(image, 0)).logits
            else:
                est_label = tf.squeeze(model(tf.expand_dims(image, 0)))
        except Exception:
            if MODEL == "segformer":
                est_label = model(tf.expand_dims(image[:, :, 0], 0)).logits
            else:
                est_label = tf.squeeze(model(tf.expand_dims(image[:, :, 0], 0)))

        if TESTTIMEAUG:
            # return the flipped prediction
            if MODEL == "segformer":
                est_label2 = np.flipud(
                    model(tf.expand_dims(np.flipud(image), 0)).logits
                )
            else:
                est_label2 = np.flipud(
                    tf.squeeze(model(tf.expand_dims(np.flipud(image), 0)))
                )
            if MODEL == "segformer":
                est_label3 = np.fliplr(
                    model(tf.expand_dims(np.fliplr(image), 0)).logits
                )
            else:
                est_label3 = np.fliplr(
                    tf.squeeze(model(tf.expand_dims(np.fliplr(image), 0)))
                )
            if MODEL == "segformer":
                est_label4 = np.flipud(
                    np.fliplr(
                        tf.squeeze(
                            model(tf.expand_dims(np.flipud(np.fliplr(image)), 0)).logits
                        )
                    )
                )
            else:
                est_label4 = np.flipud(
                    np.fliplr(
                        tf.squeeze(
                            model(tf.expand_dims(np.flipud(np.fliplr(image)), 0))
                        )
                    )
                )

            # soft voting - sum the softmax scores to return the new TTA estimated softmax scores
            est_label = est_label + est_label2 + est_label3 + est_label4

        K.clear_session()

    # heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)
    return est_label, counter


def do_seg(
    f,
    M,
    metadatadict,
    MODEL,
    sample_direc,
    NCLASSES,
    N_DATA_BANDS,
    TARGET_SIZE,
    TESTTIMEAUG,
    WRITE_MODELMETADATA,
    OTSU_THRESHOLD,
    out_dir_name="out",
    profile="minimal",
    apply_smooth=False,
):
    if profile == "meta":
        WRITE_MODELMETADATA = True
    if profile == "full":
        WRITE_MODELMETADATA = True

    # Mc = compile_models(M, MODEL)

    if f.endswith("jpg"):
        segfile = f.replace(".jpg", "_predseg.png")
    elif f.endswith("png"):
        segfile = f.replace(".png", "_predseg.png")
    elif f.endswith("npz"):  # in f:
        segfile = f.replace(".npz", "_predseg.png")

    if WRITE_MODELMETADATA:
        metadatadict["input_file"] = f

    # directory to hold the outputs of the models is named 'out' by default
    # create a directory to hold the outputs of the models, by default name it 'out' or the model name if it exists in metadatadict
    out_dir_path = os.path.normpath(sample_direc + os.sep + out_dir_name)
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    segfile = os.path.normpath(segfile)
    segfile = segfile.replace(
        os.path.normpath(sample_direc),
        os.path.normpath(sample_direc + os.sep + out_dir_name),
    )

    if WRITE_MODELMETADATA:
        metadatadict["nclasses"] = NCLASSES
        metadatadict["n_data_bands"] = N_DATA_BANDS

    if NCLASSES == 2:
        image, w, h, bigimage = get_image(
            f,
            N_DATA_BANDS,
            TARGET_SIZE,
            MODEL,
            smooth_fn=median_smooth if apply_smooth else None,
        )

        if np.std(image) == 0:
            print("Image {} is empty".format(f))
            e0 = np.zeros((w, h))
            e1 = np.zeros((w, h))

        else:
            E0, E1 = est_label_binary(
                image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE, w, h
            )

            e0 = np.average(np.dstack(E0), axis=-1)

            # del E0

            e1 = np.average(np.dstack(E1), axis=-1)
            # del E1

        est_label = (e1 + (1 - e0)) / 2

        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = np.dstack((e0, e1))
        # del e0, e1

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        if OTSU_THRESHOLD:
            thres = threshold_otsu(est_label)
            # print("Class threshold: %f" % (thres))
            est_label = (est_label > thres).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = thres

        else:
            est_label = (est_label > 0.5).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = 0.5

    else:  ###NCLASSES>2
        image, w, h, bigimage = get_image(
            f,
            N_DATA_BANDS,
            TARGET_SIZE,
            MODEL,
            smooth_fn=median_smooth if apply_smooth else None,
        )

        if np.std(image) == 0:
            print("Image {} is empty".format(f))
            est_label = np.zeros((w, h))

        else:
            est_label, counter = est_label_multiclass(
                image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE
            )

            est_label /= counter + 1
            # est_label cannot be float16 so convert to float32
            est_label = est_label.numpy().astype("float32")

            if MODEL == "segformer":
                est_label = resize(
                    est_label,
                    (1, NCLASSES, TARGET_SIZE[0], TARGET_SIZE[1]),
                    preserve_range=True,
                    clip=True,
                ).squeeze()
                est_label = np.transpose(est_label, (1, 2, 0))
                est_label = resize(est_label, (w, h))
            else:
                est_label = resize(est_label, (w, h))

        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = est_label.copy()  # np.dstack((e0,e1))

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        if np.std(image) > 0:
            est_label = np.argmax(softmax_scores, -1)
        else:
            est_label = est_label.astype("uint8")

    class_label_colormap = [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
        "#316395",
        "#ffe4e1",
        "#ff7373",
        "#666666",
        "#c0c0c0",
        "#66cdaa",
        "#afeeee",
        "#0e2f44",
        "#420420",
        "#794044",
        "#3399ff",
    ]

    class_label_colormap = class_label_colormap[:NCLASSES]

    if WRITE_MODELMETADATA:
        metadatadict["color_segmentation_output"] = segfile

    # Ensure bigimage is a NumPy array
    if hasattr(bigimage, "numpy"):
        bigimage = bigimage.numpy()

    # Try progressively simpler masks
    masks_to_try = [
        lambda img: img[:, :, 0] == 0,  # mask for first channel
        lambda img: img == 0,  # mask for all channels
    ]

    for mask_func in masks_to_try:
        try:
            color_label = label_to_colors(
                est_label,
                mask_func(bigimage),
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )
            break  # Success, exit loop
        except Exception:
            continue
    else:
        raise RuntimeError("Failed to compute color_label with all mask strategies.")

    imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)

    if WRITE_MODELMETADATA:
        metadatadict["color_segmentation_output"] = segfile

    segfile = segfile.replace("_predseg.png", "_res.npz")

    if WRITE_MODELMETADATA:
        metadatadict["grey_label"] = est_label
        np.savez_compressed(segfile, **metadatadict)

    if profile == "full":  # (profile !='minimal') and (profile !='meta'):
        #### plot overlay
        segfile = segfile.replace("_res.npz", "_overlay.png")

        if N_DATA_BANDS <= 3:
            plt.imshow(bigimage, cmap="gray")
        else:
            plt.imshow(bigimage[:, :, :3])

        plt.imshow(color_label, alpha=0.5)
        plt.axis("off")
        plt.savefig(segfile, dpi=200, bbox_inches="tight")
        plt.close("all")

        #### image - overlay side by side
        segfile = segfile.replace("_res.npz", "_image_overlay.png")

        plt.subplot(121)
        if N_DATA_BANDS <= 3:
            plt.imshow(bigimage, cmap="gray")
        else:
            plt.imshow(bigimage[:, :, :3])
        plt.axis("off")

        plt.subplot(122)
        if N_DATA_BANDS <= 3:
            plt.imshow(bigimage, cmap="gray")
        else:
            plt.imshow(bigimage[:, :, :3])
        plt.imshow(color_label, alpha=0.5)
        plt.axis("off")
        plt.savefig(segfile, dpi=200, bbox_inches="tight")
        plt.close("all")

    if profile == "full":  # (profile !='minimal') and (profile !='meta'):
        #### plot overlay of per-class probabilities
        for kclass in range(softmax_scores.shape[-1]):
            tmpfile = segfile.replace(
                "_overlay.png", "_overlay_" + str(kclass) + "prob.png"
            )

            if N_DATA_BANDS <= 3:
                plt.imshow(bigimage, cmap="gray")
            else:
                plt.imshow(bigimage[:, :, :3])

            plt.imshow(softmax_scores[:, :, kclass], alpha=0.5, vmax=1, vmin=0)
            plt.axis("off")
            plt.colorbar()
            plt.savefig(tmpfile, dpi=200, bbox_inches="tight")
            plt.close("all")


def median_smooth(img, size=15):
    """
    Apply median filter per channel if image is 3D, else apply to 2D.
    """
    if img.ndim == 3:
        return np.stack(
            [
                scipy.ndimage.median_filter(img[..., i], size=size)
                for i in range(img.shape[2])
            ],
            axis=-1,
        )
    return scipy.ndimage.median_filter(img, size=size)


def load_image(f, N_DATA_BANDS):
    """
    Load an image from file. JPG if <=3 bands, NPZ otherwise.
    Returns a NumPy uint8 array.
    """
    if N_DATA_BANDS <= 3:
        return imread(f)  # .astype('uint8')
    else:  # if N_DATA_BANDS > 3
        with np.load(f) as data:
            return data["arr_0"].astype("uint8")


def resize_and_cast(image: np.ndarray, target_size) -> tf.Tensor:
    """
    Resize image to target size, preserving range and clipping.
    Returns a TensorFlow uint8 tensor that can be used for prediction.
    This function is used to resize images before passing them to a model for prediction.

    Args:
        image (np.ndarray): Input image to resize.
        target_size (tuple): Target size for resizing (height, width).

    Returns:
        tf.Tensor: Resized image tensor.

    """
    resized = resize(
        image, (target_size[0], target_size[1]), preserve_range=True, clip=True
    )
    return tf.cast(np.array(resized), tf.uint8)


def get_image(f, N_DATA_BANDS, TARGET_SIZE, MODEL, smooth_fn=None):
    """
    Load, optionally smooth, and prepare an image for a segmentation model.
    """
    # Load image (3-band JPG or ND npz)
    bigimage = load_image(f, N_DATA_BANDS)

    # Optionally apply smoothing
    if smooth_fn is not None:
        bigimage = smooth_fn(bigimage)

    # Resize to model input size
    smallimage = resize_and_cast(bigimage, TARGET_SIZE)

    # Crop extra channels if necessary
    if N_DATA_BANDS <= 3:
        if smallimage.ndim == 3 and smallimage.shape[-1] > 3:
            smallimage = smallimage[:, :, :3]
        if bigimage.ndim == 3 and bigimage.shape[-1] > 3:
            bigimage = bigimage[:, :, :3]

    # Standardize and format for model
    image = standardize(smallimage.numpy()).squeeze()

    if MODEL == "segformer":
        # Create a 4D tensor with shape (1,3,TARGET_SIZE[0], TARGET_SIZE[1])
        if np.ndim(image) == 2:
            image = np.dstack((image, image, image))
        image = tf.transpose(image, (2, 0, 1))

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return image, w, h, bigimage
