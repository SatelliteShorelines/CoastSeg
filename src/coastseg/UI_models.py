# standard python imports
import os
import glob
import logging
# internal python imports
from coastseg import common
from coastseg import zoo_model
from coastseg.tkinter_window_creator import Tkinter_Window_Creator

# external python imports
import ipywidgets
from IPython.display import display
from tkinter import filedialog
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML
from ipywidgets import RadioButtons
from ipywidgets import Output
from ipywidgets import Checkbox
from tkinter import messagebox

logger = logging.getLogger(__name__)

class UI_Models:
    # all instances of UI will share the same debug_view
    model_view = Output(layout={"border": "1px solid black"})
    run_model_view = Output(layout={"border": "1px solid black"})

    def __init__(self):
        # Controls size of ROIs generated on map
        self.model_dict = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "ENSEMBLE",
            "model_type": "s2-landsat78-4class_6950472",
            "otsu":False,
            "tta":False,
        }
        # list of RGB and MNDWI models available
        self.RGB_models = [
            "s2-landsat78-4class_6950472",
            "sat_RGB_2class_7384255",
            "sat_RGB_4class_6950472",
        ]
        self.five_band_models = [
            "sat-5band-4class_7344606",
        ]
        self.MNDWI_models = ["s2-landsat78-4class_7352850"]
        self.NDWI_models = ["s2-landsat78-4class_7352859"]
        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()

    def create_dashboard(self):
        model_choices_box = HBox(
            [self.model_input_dropdown, 
            self.model_dropdown,
            self.model_implementation]
        )
        checkboxes = HBox([self.GPU_dropdown,self.otsu_radio,self.tta_radio])
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

        self.otsu_radio = RadioButtons(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Otsu Threshold:",
            disabled=False,
        )
        self.otsu_radio.observe(self.handle_otsu, "value")

        self.tta_radio = RadioButtons(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Test Time Augmentation:",
            disabled=False,
        )
        self.tta_radio.observe(self.handle_tta, "value")

        self.model_input_dropdown = ipywidgets.RadioButtons(
            options=["RGB", "MNDWI", "NDWI", "5 Bands"],
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

        # allow user to select number of GPUs
        self.GPU_dropdown =ipywidgets.IntSlider(
            value=0,
            min=0,
            max=5,
            step=1,
            description='GPU(s):',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.GPU_dropdown.observe(self.handle_GPU_dropdown, "value")

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
                <br> - This will open a pop up window where the RGB folder must be selected.<br>\
                    - The model will be applied to the 'model input' folder selected and the model outputs will be generated within a subdirectory\
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


    def handle_model_type(self, change):
        self.model_dict["model_type"] = change["new"]

    def handle_GPU_dropdown(self, change):
        if change["new"] == 0:
            self.model_dict["use_GPU"] = "0"
        else:
            self.model_dict["use_GPU"] = str(change["new"])

    def handle_otsu(self,change):
        if change["new"] == "Enabled":
            self.model_dict["otsu"]=True
        if change["new"] == "Disabled":
            self.model_dict["otsu"]=False

    def handle_tta(self,change):
        if change["new"] == "Enabled":
            self.model_dict["tta"]=True
        if change["new"] == "Disabled":
            self.model_dict["tta"]=False

    def handle_model_input_change(self, change):
        if change["new"] == "RGB":
            self.model_dropdown.options = self.RGB_models
        if change["new"] == "MNDWI":
            self.model_dropdown.options = self.MNDWI_models
        if change["new"] == "NDWI":
            self.model_dropdown.options = self.NDWI_models
        if change["new"] == "5 Bands":
            self.model_dropdown.options = self.five_band_models

    @model_view.capture(clear_output=True)
    def use_data_button_clicked(self, button: "ipywidgets.Button") -> None:
        """Runs on use data button clicked
        Copies all jpgs from data directory to new directory
        called segmentation_data. imagery in segmentation data will be used as inputs
        for models.
        Args:
            button (ipywidgets.Button): instance of button
        """
        print("Loading in jpgs from  data directory")
        sample_direc = common.get_jpgs_from_data()
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
            # gets GPU or CPU depending on whether use_GPU is True
            zoo_model.get_GPU(self.model_dict["use_GPU"])
            # Disable run and open results buttons while the model is running
            self.open_results_button.disabled = True
            self.run_model_button.disabled = True
            model_choice = self.model_dict["implementation"]
            zoo_model_instance = zoo_model.Zoo_Model()

            logger.info(
                f"\nolder selected directory of RGBs: {self.model_dict['sample_direc']}\n"
            )
            # get path to RGB directory for models
            # all other necessary files are relative to RGB directory
            RGB_path = common.get_RGB_in_path(self.model_dict["sample_direc"])
            self.model_dict["sample_direc"] = RGB_path
            print(
                f"current selected directory of RGBs: {self.model_dict['sample_direc']}"
            )

            # convert RGB to MNDWI or NDWI
            output_type = self.model_input_dropdown.value
            print(f"Selected output type: {output_type}")
            if output_type in ["MNDWI", "NDWI"]:
                RGB_path = self.model_dict["sample_direc"]
                output_path = os.path.dirname(RGB_path)
                # default filetype is NIR and if NDWI is selected else filetype to SWIR
                filetype =  "NIR" if output_type == "NDWI" else "SWIR"
                infrared_path = os.path.join(output_path, filetype)
                zoo_model.RGB_to_infrared(
                    RGB_path, infrared_path, output_path, output_type
                )
                # newly created imagery (NDWI or MNDWI) is located at output_path 
                output_path = os.path.join(output_path, output_type)
                # set sample_direc to hold location of NDWI imagery
                self.model_dict["sample_direc"] = output_path
                print(f"Model outputs will be saved to {output_path}")
            elif output_type in ['5 Bands']:
                RGB_path = self.model_dict["sample_direc"]
                output_path = os.path.dirname(RGB_path)
                NIR_path = os.path.join(output_path, "NIR")
                NDWI_path  = zoo_model.RGB_to_infrared(
                    RGB_path, NIR_path, output_path, "NDWI"
                )
                SWIR_path = os.path.join(output_path, "SWIR")
                MNDWI_path  = zoo_model.RGB_to_infrared(
                    RGB_path, SWIR_path, output_path, "MNDWI"
                )
                five_band_path=os.path.join(output_path, "five_band")
                if not os.path.exists(five_band_path):
                    os.mkdir(five_band_path)
                five_band_path = zoo_model.get_five_band_imagery(RGB_path,MNDWI_path,NDWI_path,five_band_path)
                # set sample_direc to hold location of NDWI imagery
                self.model_dict["sample_direc"] = five_band_path
                print(f"Model outputs will be saved to {five_band_path}")


            # specify dataset_id to download selected model
            dataset_id = self.model_dict["model_type"]
            use_otsu = self.model_dict["otsu"]
            use_tta = self.model_dict["tta"]
            # First download the specified model
            zoo_model_instance.download_model(model_choice, dataset_id)
            # Get weights as list
            weights_list = zoo_model_instance.get_weights_list(model_choice)
            # Load the model from the config files
            model, model_list, config_files, model_types = zoo_model_instance.get_model(
                weights_list
            )
            metadatadict = zoo_model_instance.get_metadatadict(
                weights_list, config_files, model_types
            )
            # # Compute the segmentation
            zoo_model_instance.compute_segmentation(
                self.model_dict["sample_direc"],
                model_list,
                metadatadict,
                use_tta,
                use_otsu
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
