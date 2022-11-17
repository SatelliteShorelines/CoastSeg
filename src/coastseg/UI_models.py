# standard python imports
import os
import glob
import json

# internal python imports
from coastseg import common
from coastseg.zoo_model import Zoo_Model
from coastseg.tkinter_window_creator import Tkinter_Window_Creator

# external python imports
import ipywidgets
import geopandas as gpd
from ipyleaflet import GeoJSON
from IPython.display import display, clear_output
from tkinter import Tk, filedialog
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML
from ipywidgets import RadioButtons
from ipywidgets import SelectMultiple
from ipywidgets import Output
from ipywidgets import Checkbox
from tkinter import messagebox


class UI_Models:
    # all instances of UI will share the same debug_view
    model_view = Output(layout={"border": "1px solid black"})
    run_model_view = Output(layout={"border": "1px solid black"})
    GPU_view = Output()

    def __init__(self):
        # Controls size of ROIs generated on map
        self.model_dict = {
            "sample_direc": None,
            "use_GPU": False,
            "use_CRF": False,
            "implementation": "ENSEMBLE",
            "model_type": "landsat_6229071",
        }
        # list of RGB and MNDWI models available
        self.RGB_models = [
            "landsat_6229071",
            "landsat_6230083",
            "SWED-RGB_6824384",
            "coast-train-RGB_6950479",
            "S2-water-SWED_6950474",
        ]
        self.MNDWI_models = ["SWED-MNDWI_6824342"]
        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()

    def create_dashboard(self):
        model_choices_box = HBox(
            [self.model_input_dropdown, self.model_dropdown, self.model_implementation]
        )
        checkboxes = HBox([self.GPU_checkbox, self.CRF_checkbox])
        instr_vbox = VBox(
            [
                self.instr_header,
                self.line_widget,
                self.instr_use_data,
                self.instr_select_images,
                self.instr_run_model,
            ]
        )
        display(
            checkboxes,
            UI_Models.GPU_view,
            model_choices_box,
            instr_vbox,
            self.use_data_button,
            self.use_select_images_button,
            self.line_widget,
            UI_Models.model_view,
            self.run_model_button,
            UI_Models.run_model_view,
            self.open_results_button,
        )

    def _create_widgets(self):
        self.model_implementation = RadioButtons(
            options=["ENSEMBLE", "BEST"],
            value="ENSEMBLE",
            description="Select:",
            disabled=False,
        )
        self.model_implementation.observe(self.handle_model_implementation, "value")

        self.model_input_dropdown = ipywidgets.RadioButtons(
            options=["RGB", "MNDWI"],
            value="RGB",
            description="Model Input:",
            disabled=False,
        )
        self.model_input_dropdown.observe(self.handle_model_input_change, names="value")

        self.model_dropdown = ipywidgets.RadioButtons(
            options=self.RGB_models,
            value=self.RGB_models[0],
            description="Select Model:",
            disabled=False,
        )
        self.model_dropdown.observe(self.handle_model_type, "value")

        self.GPU_checkbox = Checkbox(
            value=False, description="Use GPU?", disabled=False, indent=False
        )
        self.GPU_checkbox.observe(self.handle_GPU_checkbox, "value")

        self.CRF_checkbox = Checkbox(
            value=False,
            description="Use CRF post-processing",
            disabled=False,
            indent=False,
        )
        self.CRF_checkbox.observe(self.handle_CRF_checkbox, "value")

    def _create_buttons(self):
        # button styles
        load_style = dict(button_color="#69add1")
        action_style = dict(button_color="#ae3cf0")

        self.run_model_button = Button(description="Run Model", style=action_style)
        self.run_model_button.on_click(self.run_model_button_clicked)
        self.use_data_button = Button(description="Use Data Button", style=load_style)
        self.use_data_button.on_click(self.use_data_button_clicked)
        self.use_select_images_button = Button(
            description="Select Your Images", style=load_style
        )
        self.use_select_images_button.on_click(self.use_select_images_button_clicked)
        self.open_results_button = Button(
            description="Open Model Results", style=load_style
        )
        self.open_results_button.on_click(self.open_results_button_clicked)

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.line_widget = HTML(
            value="____________________________________________________"
        )

        self.instr_header = HTML(
            value="<h4>Click ONE of the following buttons:</h4>",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.instr_use_data = HTML(
            value="<b>1. Use Data Folder Button</b> \
                <br> - When CoastSat downloads imagery it created a folder called 'data'in the CoastSeg directory.\
                    The jpgs within the 'data' folder will be copied to another folder with a name such as\
                    <span style=\"background-color:LightGray;color: black;\">segmentation_data_2022-07-07__10_hr_04_min58 </span>\
                    (the date and time will be the current date and time) <br> \
                The model will be applied to this folder and the model outputs will be generated within a subdirectory \
                    called 'out'",
            layout=Layout(margin="0px 0px 0px 20px"),
        )

        self.instr_select_images = HTML(
            value="<b>2. Select Images Button</b> \
                <br> - This will open a pop up window where the folder containing the jpgs can be selected.<br>\
                    - The model will be applied to this folder and the model outputs will be generated within a subdirectory\
                    called 'out'<br>\
            - <span style=\"background-color:yellow;color: black;\">WARNING :</span> You will not be able to see the files within the folder you select.<br>\
            ",
            layout=Layout(margin="0px 0px 0px 20px"),
        )

        self.instr_run_model = HTML(
            value="<b>3. Run Model Button</b> \
                <br> - Make sure to click Select Images Button or Use Data Button.<br>\
                    - The model will be applied to the selected folder and the model outputs will be generated within a subdirectory\
                    called 'out'<br>\
            - <span style=\"background-color:yellow;color: black;\">WARNING :</span> You should not run multiple models on the same folder. Otherwise not all the model outputs\
            will be saved to the folder.<br>\
            ",
            layout=Layout(margin="0px 0px 0px 20px"),
        )

    def handle_model_implementation(self, change):
        self.model_dict["implementation"] = change["new"]

    @GPU_view.capture(clear_output=True)
    def handle_GPU_checkbox(self, change):
        if change["new"] == True:
            self.model_dict["use_GPU"] = True
            print("Using the GPU")
        else:
            self.model_dict["use_GPU"] = False
            print("Not using the GPU")

    @GPU_view.capture(clear_output=True)
    def handle_CRF_checkbox(self, change):
        if change["new"] == True:
            self.model_dict["use_CRF"] = True
            print("Using CRF post-processing")
        else:
            self.model_dict["use_CRF"] = False
            print("Not using CRF post-processing")

    def handle_model_type(self, change):
        self.model_dict["model_type"] = change["new"]

    def handle_model_input_change(self, change):
        if change["new"] == "MNDWI":
            self.model_dropdown.options = self.MNDWI_models
        if change["new"] == "RGB":
            self.model_dropdown.options = self.RGB_models

    @model_view.capture(clear_output=True)
    def use_data_button_clicked(self, button):
        # Use the data folder as the input for segmentation
        print("Loading in the jpgs from the data directory")
        # Copy the jpgs from data to a new folder called segmentation_data_[datetime]
        if "MNDWI" in self.model_dropdown.value:
            sample_direc = r"C:\1_USGS\CoastSeg\repos\2_CoastSeg\CoastSeg_fork\Seg2Map\MNDWI_outputs\MNDWI_ouputs_2022-07-21__07_hr_57_min14"
            # sample_direc = common.get_jpgs_from_data('MNDWI')
            # RGB_path=sample_direc+os.sep+'RGB'
            # NIR_path=sample_direc+os.sep+'NIR'
            # sample_direc = coastseg_map.RGB_to_MNDWI(RGB_path,NIR_path,sample_direc)
            self.model_dict["sample_direc"] = sample_direc
        else:
            sample_direc = common.get_jpgs_from_data("RGB")
            jpgs = glob.glob1(sample_direc + os.sep, "*jpg")
            if jpgs == []:
                with Tkinter_Window_Creator():
                    messagebox.showinfo(
                        "No JPGs found",
                        "\nERROR!\nThe directory contains no jpgs! Please select a directory with jpgs.",
                    )
            elif jpgs != []:
                self.model_dict["sample_direc"] = sample_direc
        print(f"\nContents of the data directory saved in {sample_direc}")

    @run_model_view.capture(clear_output=True)
    def run_model_button_clicked(self, button):
        print("Called Run Model")
        if self.model_dict["sample_direc"] is None:
            with Tkinter_Window_Creator():
                messagebox.showinfo(
                    "", "You must click 'Use Data' or 'Select Images' First"
                )
            return
        else:
            if self.model_dict["use_GPU"] == False:
                print("Not using the GPU")
                ## to use the CPU (not recommended):
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            elif self.model_dict["use_GPU"] == True:
                print("Using the GPU")
                # use the first available GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #'1'

            # Disable run and open results buttons while the model is running
            self.open_results_button.disabled = True
            self.run_model_button.disabled = True
            # self.model_dict['implementation']=model_implementation.value
            model_choice = self.model_dict["implementation"]
            zoo_model = Zoo_Model()
            # specify dataset_id as well as data type to download selected model
            dataset = "MNDWI" if "MNDWI" in self.model_dropdown.value else "RGB"
            dataset_id = self.model_dict["model_type"]
            # First download the specified model
            zoo_model.download_model(dataset, dataset_id)
            # Get weights as list
            Ww = zoo_model.get_weights_list(model_choice)
            # Load the model from the config files
            model, model_list, config_files, model_types = zoo_model.get_model(Ww)
            metadatadict = zoo_model.get_metadatadict(Ww, config_files, model_types)
            # # Compute the segmentation
            zoo_model.compute_segmentation(
                self.model_dict["sample_direc"],
                model_list,
                metadatadict,
                self.model_dict["use_CRF"],
            )
            # Enable run and open results buttons when model has executed
            self.run_model_button.disabled = False
            self.open_results_button.disabled = False

    @run_model_view.capture(clear_output=True)
    def open_results_button_clicked(self, button):
        """open_results_button_clicked on click handler for 'open results' button.

        Opens a tkinter window that shows the model outputs in the folder.

        Args:
            button (Button): button that was clicked

        Raises:
            FileNotFoundError: raised when the directory where the model outputs are saved does not exist
        """
        with Tkinter_Window_Creator():
            if self.model_dict["sample_direc"] is None:
                messagebox.showinfo("", "You must click 'Run Model' first")
            else:
                # path to directory containing model outputs
                model_results_path = os.path.abspath(self.model_dict["sample_direc"])
                if not os.path.exists(model_results_path):
                    messagebox.showerror(
                        "File Not Found",
                        "The directory for the model outputs could not be found",
                    )
                    raise FileNotFoundError
                filedialog.askopenfiles(
                    initialdir=model_results_path, title="Open Results"
                )

    @model_view.capture(clear_output=True)
    def use_select_images_button_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            # path to data directory containing data downloaded using coastsat
            data_path = os.path.join(os.getcwd(), "data")
            if os.path.exists(data_path):
                data_path = os.path.join(os.getcwd(), "data")
            else:
                data_path = os.getcwd()
            tk_root.filename = filedialog.askdirectory(
                initialdir=data_path,
                title="Select directory of images",
            )
            # Save the filename as an attribute of the button
            if tk_root.filename:
                sample_direc = tk_root.filename
                print(f"The images in the folder will be segmented :\n{sample_direc} ")
                jpgs = glob.glob1(sample_direc + os.sep, "*jpg")
                if jpgs == []:
                    messagebox.showerror(
                        "No JPGs Found",
                        "\nERROR!\nThe directory contains no jpgs! Please select a directory with jpgs.",
                    )
                elif jpgs != []:
                    self.model_dict["sample_direc"] = sample_direc
            else:
                messagebox.showerror(
                    "Invalid directory", "You must select a valid directory first!"
                )
