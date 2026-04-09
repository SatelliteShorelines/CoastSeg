# How to Run Zoo Models

Use this guide to run CoastSeg zoo models on imagery you already downloaded.

!!! note
	Before you start:

	- Download your imagery from Google Earth Engine first.
	- The zoo workflow processes one region of interest (ROI) at a time.
	- Install the zoo workflow first by following [Install the Zoo Workflow](install-zoo-workflow.md).

## Phase 1: Download a Model

Before you can run the zoo workflow, you need to download at least one model.

1. Activate your main CoastSeg environment.

	```bash
	conda activate coastseg
	```

2. Open the notebook `SDS_zoo_classifier.ipynb`.
3. Use the download button in the notebook to download a model.

	- You can also download a model with the script `download_zoo_model.py`. If you want to use the script instead of the notebook, see [How to Download Models](how-to-download-zoo-models.md).

4. Confirm that the model folder was saved in `CoastSeg/models`.

## Phase 2: Activate the Zoo Environment

Choose one setup method.

### Option 1: Pixi (Recommended)

```bash
cd <coastseg_location>
cd segmentation_workflow
pixi shell
```

On the first run, `pixi shell` installs the environment and then activates it.

You will know the environment is active when you see `(segmentation_workflow)` in the terminal.

### Option 2: Conda

```bash
cd <coastseg_location>
cd segmentation_workflow
conda activate segmentation_workflow
```

## Phase 3: Run the Models

Before running the command below, make sure:

- The `(segmentation_workflow)` environment is active
- You are inside the `segmentation_workflow` folder

### Command

```bash
python run_zoo_segmentation_models.py \
	-i "<input_dir>" \
	-o "<output_dir>" \
	-m "<model_dir>" \
	[--implementation BEST|ENSEMBLE] \
	[--overwrite] \
	[--cpu-only]
```

### What the command means

- `-i`, `--input-dir`: Folder containing the input images. These can be `jpg`, `png`, or `tif` files.
- `-o`, `--output-dir`: Folder where the model outputs will be saved.
- `-m`, `--model`: Folder containing the model files, including the `.h5` weights and `.json` config files.

### Default behavior

- `--implementation BEST` is used unless you choose something else
- A GPU is used automatically if one is available

### Example

```bash
python run_zoo_segmentation_models.py \
	-i "C:\path\to\jpg_files" \
	-o "C:\path\to\predictions" \
	-m "C:\path\to\model_dir"
```

### Real example

```bash
python run_zoo_segmentation_models.py \
	-i "C:\CoastSeg\data\ID_rnv2_datetime01-16-26__03_50_41\jpg_files\preprocessed\RGB" \
	-o "C:\CoastSeg\sessions\model_outputs_session" \
	-m "C:\CoastSeg\models\global_segformer_RGB_4class_14036903"
```

This example runs the model `global_segformer_RGB_4class_14036903` on the images in `C:\CoastSeg\data\ID_rnv2_datetime01-16-26__03_50_41\jpg_files\preprocessed\RGB` and saves the results to `C:\CoastSeg\sessions\model_outputs_session`.

## Outputs

All results are written to `<output_dir>`.

For each input image, the workflow creates:

- `<image_stem>_predseg.png`: A colorized segmentation mask
- `<image_stem>_res.npz`: A compressed NumPy file containing the predicted labels and metadata

It also creates these summary files for the full run:

- `model_settings.json`: The settings used for the run, including implementation mode, model type, and GPU usage
- `model_info.json`: Information about the model, including the model directory, class names, and water-class indices
- `segmentation_summary.json`: A summary of the run, including processed images, output files, skipped files, and failures

The output folder keeps the same relative folder structure as the input folder.

## Phase 4: Extract Shorelines from the Segmentations

After the model finishes, you can extract shorelines from the segmentation outputs.

To do this, use the script `3_zoo_workflow_extract_shorelines.py` in the main CoastSeg folder.
Or you can use the `SDS_zoo_classifier.ipynb` and use the `workflow b` option to extract shorelines from the folder of segmentations you just created.

1. Exit or deactivate the `segmentation_workflow` environment.
2. Go back to the main CoastSeg folder.

	```bash
	cd ..
	```

3. Activate your main CoastSeg environment.

	Use one of these options:

	```bash
	conda activate CoastSeg
	```

	or

	```bash
	pixi shell
	```

4. Open `3_zoo_workflow_extract_shorelines.py` in a code editor such as VS Code.
	- Read the instructions in the script for a guide on how to use it.
5. Update the script so it points to the folder containing your segmentation outputs.
6. Run the script.