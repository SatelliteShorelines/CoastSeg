# standard python imports
import os
import glob
import logging

# internal python imports
from coastseg import common
from coastseg import zoo_model

# external python imports
import ipywidgets
from IPython.display import display
from ipywidgets import Button
from ipywidgets import ToggleButton
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML
from ipywidgets import RadioButtons
from ipywidgets import Output
from ipyfilechooser import FileChooser

# icons sourced from https://fontawesome.com/v4/icons/

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
            "implementation": "BEST",
            "model_type": "sat_RGB_4class_6950472",
            "otsu": False,
            "tta": False,
        }
        # list of RGB and MNDWI models available
        self.RGB_models = [
            "sat_RGB_2class_7448405",
            "sat_RGB_4class_6950472",
        ]
        self.five_band_models = [
            "sat_5band_4class_7344606",
            "sat_5band_2class_7448390",
        ]
        self.MNDWI_models = [
            "sat_MNDWI_4class_7352850",
            "sat_MNDWI_2class_7557080",
        ]
        self.NDWI_models = [
            "sat_NDWI_4class_7352859",
            "sat_NDWI_2class_7557072",
        ]

        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()

    def create_dashboard(self):
        model_choices_box = HBox(
            [self.model_input_dropdown, self.model_dropdown, self.model_implementation]
        )
        checkboxes = HBox([self.otsu_radio, self.tta_radio])
        instr_vbox = VBox(
            [
                self.instr_header,
                self.line_widget,
                self.instr_select_images,
                self.instr_run_model,
            ]
        )
        self.file_row = HBox([])
        self.warning_row = HBox([])
        display(
            checkboxes,
            model_choices_box,
            instr_vbox,
            self.use_select_images_button,
            self.line_widget,
            self.warning_row,
            self.file_row,
            UI_Models.model_view,
            self.run_model_button,
            UI_Models.run_model_view,
            self.open_results_button,
        )

    def _create_widgets(self):
        self.model_implementation = RadioButtons(
            options=["ENSEMBLE", "BEST"],
            value="BEST",
            description="Select:",
            disabled=False,
        )
        self.model_implementation.observe(self.handle_model_implementation, "value")

        self.otsu_radio = RadioButtons(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Otsu Threshold:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.otsu_radio.observe(self.handle_otsu, "value")

        self.tta_radio = RadioButtons(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Test Time Augmentation:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.tta_radio.observe(self.handle_tta, "value")

        self.model_input_dropdown = ipywidgets.RadioButtons(
            options=["RGB", "MNDWI", "NDWI", "RGB+MNDWI+NDWI"],
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

    def _create_buttons(self):
        # button styles
        load_style = dict(button_color="#69add1")
        action_style = dict(button_color="#ae3cf0")

        self.run_model_button = Button(
            description="Run Model",
            style=action_style,
            icon="fa-bolt",
        )
        self.run_model_button.on_click(self.run_model_button_clicked)

        self.use_select_images_button = Button(
            description="Select Images",
            style=load_style,
            icon="fa-file-image-o",
        )
        self.use_select_images_button.on_click(self.use_select_images_button_clicked)
        self.open_results_button = Button(
            description="Open Results",
            style=load_style,
            icon="folder-open-o",
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

        self.instr_select_images = HTML(
            value="<b>1. Select Images Button</b> \
                <br> - This will open a pop up window where the RGB folder must be selected.<br>\
                    - The model will be applied to the 'model input' folder selected and the model outputs will be generated within a subdirectory\
                    called 'out'<br>\
            - <span style=\"background-color:yellow;color: black;\">WARNING :</span> You will not be able to see the files within the folder you select.<br>\
            ",
            layout=Layout(margin="0px 0px 0px 20px"),
        )

        self.instr_run_model = HTML(
            value="<b>2. Run Model Button</b> \
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
        # 2 class model has not been selected disable otsu threhold
        if "2class" not in change["new"]:
            if self.otsu_radio.value == "Enabled":
                self.model_dict["otsu"] = False
                self.otsu_radio.value = "Disabled"
            self.otsu_radio.disabled = True
        # 2 class model was selected enable otsu threhold radio button
        if "2class" in change["new"]:
            self.otsu_radio.disabled = False

        logger.info(f"change: {change}")
        self.model_dict["model_type"] = change["new"]

    def handle_otsu(self, change):
        if change["new"] == "Enabled":
            self.model_dict["otsu"] = True
        if change["new"] == "Disabled":
            self.model_dict["otsu"] = False

    def handle_tta(self, change):
        if change["new"] == "Enabled":
            self.model_dict["tta"] = True
        if change["new"] == "Disabled":
            self.model_dict["tta"] = False

    def handle_model_input_change(self, change):
        if change["new"] == "RGB":
            self.model_dropdown.options = self.RGB_models
        if change["new"] == "MNDWI":
            self.model_dropdown.options = self.MNDWI_models
        if change["new"] == "NDWI":
            self.model_dropdown.options = self.NDWI_models
        if change["new"] == "RGB+MNDWI+NDWI":
            self.model_dropdown.options = self.five_band_models

    @run_model_view.capture(clear_output=True)
    def run_model_button_clicked(self, button):
        if self.model_dict["sample_direc"] is None:
            self.launch_error_box(
                "Cannot Run Model",
                "You must click 'Select Images' first",
            )
            return
        print("Running the model. Please wait.")
        model_implementation = self.model_dict["implementation"]
        model_name = self.model_dict["model_type"]
        use_otsu = self.model_dict["otsu"]
        use_tta = self.model_dict["tta"]

        zoo_model_instance = zoo_model.Zoo_Model()

        # get full path to directory named 'RGB' containing RGBs
        RGB_path = common.get_RGB_in_path(self.model_dict["sample_direc"])
        self.model_dict["sample_direc"] = RGB_path

        # convert RGB to MNDWI, NDWI,or 5 band
        output_type = self.model_input_dropdown.value
        logger.info(f"output_type: {output_type}")
        if output_type != "RGB":
            self.model_dict["sample_direc"] = zoo_model.get_imagery_directory(
                output_type, RGB_path
            )

        print(f"Model outputs will be saved to {self.model_dict['sample_direc']}")

        zoo_model_instance.run_model(
            model_implementation,
            self.model_dict["sample_direc"],
            model_name=model_name,
            use_GPU="0",
            use_otsu=use_otsu,
            use_tta=use_tta,
        )

    @run_model_view.capture(clear_output=True)
    def open_results_button_clicked(self, button):
        """open_results_button_clicked on click handler for 'open results' button.

        prints location of model outputs

        Args:
            button (Button): button that was clicked

        Raises:
            FileNotFoundError: raised when the directory where the model outputs are saved does not exist
        """
        if self.model_dict["sample_direc"] is None:
            self.launch_error_box(
                "Cannot Open Results", "You must click 'Run Model' first"
            )
        else:
            # path to directory containing model outputs
            model_results_path = os.path.abspath(self.model_dict["sample_direc"])
            if not os.path.exists(model_results_path):
                self.launch_error_box(
                    "File Not Found",
                    "The directory for the model outputs could not be found",
                )
                raise FileNotFoundError
            else:
                print(f"Model outputs located at:\n{model_results_path}")

    @model_view.capture(clear_output=True)
    def load_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            sample_direc = os.path.abspath(filechooser.selected)
            print(f"The images in the folder will be segmented :\n{sample_direc} ")
            jpgs = glob.glob1(sample_direc + os.sep, "*jpg")
            if jpgs == []:
                self.launch_error_box(
                    "File Not Found",
                    "The directory contains no jpgs! Please select a directory with jpgs.",
                )
            elif jpgs != []:
                self.model_dict["sample_direc"] = sample_direc

    @model_view.capture(clear_output=True)
    def use_select_images_button_clicked(self, button):
        # Prompt the user to select a directory of images
        file_chooser = common.create_dir_chooser(
            self.load_callback, title="Select directory of images"
        )
        # clear row and close all widgets in self.file_row before adding new file_chooser
        common.clear_row(self.file_row)
        # add instance of file_chooser to self.file_row
        self.file_row.children = [file_chooser]

    def launch_error_box(self, title: str = None, msg: str = None):
        # Show user error message
        warning_box = common.create_warning_box(title=title, msg=msg)
        # clear row and close all widgets in self.file_row before adding new warning_box
        common.clear_row(self.warning_row)
        # add instance of warning_box to self.warning_row
        self.warning_row.children = [warning_box]
