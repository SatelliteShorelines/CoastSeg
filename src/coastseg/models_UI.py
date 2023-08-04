# standard python imports
import os
import glob
import logging

# internal python imports
from coastseg import zoo_model
from coastseg import settings_UI
from coastseg import common
from coastseg.upload_feature_widget import FileUploader

# external python imports
import ipywidgets
from IPython.display import display
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML
from ipywidgets import RadioButtons
from ipywidgets import Output
from ipyfilechooser import FileChooser
from ipywidgets import FloatText

# icons sourced from https://fontawesome.com/v4/icons/

logger = logging.getLogger(__name__)


class UI_Models:
    # all instances of UI will share the same debug_view
    model_view = Output(layout={"border": "1px solid black"})
    extract_shorelines_view = Output(layout={"border": "1px solid black"})
    tidal_correction_view = Output(layout={"border": "1px solid black"})
    run_model_view = Output(layout={"border": "1px solid black"})

    def __init__(self):
        self.settings_dashboard = settings_UI.Settings_UI()
        self.zoo_model_instance = zoo_model.Zoo_Model()
        instructions = "Upload a GeoJSON file that contains either transects or shorelines.\
        If no file is provided, the CoastSeg will automatically attempt to load an available file for you.\
        In the event that no transects or shorelines are available within the specified region, an error will occur."
        self.fileuploader = FileUploader(
            title="",
            instructions=instructions,
            filter_pattern="*geojson",
            dropdown_options=["transects", "shorelines"],
            file_selection_title="Select a geojson file",
            max_width=400,
        )
        # Controls size of ROIs generated on map
        self.model_dict = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "BEST",
            "model_type": "segformer_RGB_4class_8190958",
            "otsu": False,
            "tta": False,
        }
        # list of RGB and MNDWI models available
        self.RGB_models = [
            "segformer_RGB_4class_8190958",
            "sat_RGB_4class_6950472",
        ]
        # self.five_band_models = [
        #     "sat_5band_4class_7344606",
        # ]
        self.MNDWI_models = [
            "segformer_MNDWI_4class_8213443",
        ]
        self.NDWI_models = [
            "segformer_NDWI_4class_8213427",
        ]
        self.session_name = ""
        self.shoreline_session_directory = ""
        self.model_session_directory = ""
        self.tides_file = ""
        self.shoreline_session_name = ""

        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()
        self.create_tidal_correction_widget()
        self.warning_row1 = HBox([])
        self.warning_row2 = HBox([])
        self.warning_row3 = HBox([])

    def get_warning_box(self, position: int = 1):
        if position == 1:
            return self.warning_row1
        if position == 2:
            return self.warning_row2
        if position == 3:
            return self.warning_row3

    def clear_extract_shorelines_btn(
        self,
    ):
        clear_button = Button(
            description="Clear TextBox", style=dict(button_color="#a3adac")
        )
        clear_button.on_click(self.clear_extract_shorelines_view)
        return clear_button

    def clear_run_model_btn(
        self,
    ):
        clear_button = Button(
            description="Clear TextBox", style=dict(button_color="#a3adac")
        )
        clear_button.on_click(self.clear_run_model_view)
        return clear_button

    def clear_tidal_correction_btn(
        self,
    ):
        clear_button = Button(
            description="Clear TextBox", style=dict(button_color="#a3adac")
        )
        clear_button.on_click(self.clear_tidal_correction_view)
        return clear_button

    def clear_run_model_view(self, btn):
        UI_Models.run_model_view.clear_output()

    def clear_extract_shorelines_view(self, btn):
        UI_Models.extract_shorelines_view.clear_output()

    def clear_tidal_correction_view(self, btn):
        UI_Models.tidal_correction_view.clear_output()

    def get_model_instance(self):
        return self.zoo_model_instance

    def set_session_name(self, name: str):
        self.session_name = str(name).strip()

    def get_session_name(
        self,
    ):
        return self.session_name

    def set_shoreline_session_name(self, name: str):
        self.shoreline_session_name = str(name).strip()

    def get_shoreline_session_name(
        self,
    ):
        return self.shoreline_session_name

    def get_shoreline_session_selection(self):
        output = Output()
        self.shoreline_session_name_text = ipywidgets.Text(
            value="",
            placeholder="Enter name",
            description="Extracted Shoreline Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = ipywidgets.Button(description="Enter")

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            session_name = str(self.shoreline_session_name_text.value).strip()
            session_path = common.create_directory(os.getcwd(), "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(f"Session {session_name} already exists at {new_session_path}")
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} will be created at {new_session_path}")
                self.set_shoreline_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.shoreline_session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def get_session_selection(self):
        output = Output()
        self.session_name_text = ipywidgets.Text(
            value="",
            placeholder="Enter a session name",
            description="Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = ipywidgets.Button(description="Enter")

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            session_name = str(self.session_name_text.value).strip()
            session_path = common.create_directory(os.getcwd(), "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(f"Session {session_name} already exists at {new_session_path}")
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} will be created at {new_session_path}")
                self.set_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def create_dashboard(self):
        model_choices_box = HBox(
            [self.model_input_dropdown, self.model_dropdown, self.model_implementation]
        )
        checkboxes = HBox([self.otsu_radio, self.tta_radio])

        # run model controls
        run_model_buttons = VBox(
            [
                self.run_model_instr,
                self.get_session_selection(),
                self.use_select_images_button,
                self.run_model_button,
            ]
        )

        # extract shorelines controls
        extract_shorelines_controls = VBox(
            [
                self.extract_shorelines_instr,
                self.get_shoreline_session_selection(),
                self.fileuploader.get_FileUploader_widget(),
                self.select_model_session_button,
                self.extract_shorelines_button,
            ]
        )
        # tidal correction controls
        tidal_correction_controls = VBox(
            [
                self.tidally_correct_instr,
                self.select_extracted_shorelines_session_button,
                self.create_tidal_correction_widget(),
            ]
        )

        self.file_row = HBox([])
        self.extracted_shoreline_file_row = HBox([])
        self.tidal_correct_file_row = HBox([])
        display(
            self.settings_dashboard.render(),
            checkboxes,
            model_choices_box,
            run_model_buttons,
            HBox([self.clear_run_model_btn(), UI_Models.run_model_view]),
            self.file_row,
            self.warning_row1,
            self.line_widget,
            extract_shorelines_controls,
            self.extracted_shoreline_file_row,
            HBox(
                [self.clear_extract_shorelines_btn(), UI_Models.extract_shorelines_view]
            ),
            self.warning_row2,
            self.line_widget,
            tidal_correction_controls,
            self.tidal_correct_file_row,
            HBox([self.clear_tidal_correction_btn(), UI_Models.tidal_correction_view]),
            self.warning_row3,
        )

    def create_tidal_correction_widget(self):
        load_style = dict(button_color="#69add1", description_width="initial")

        self.beach_slope_text = FloatText(value=0.1, description="Beach Slope")
        self.reference_elevation_text = FloatText(value=0.585, description="Elevation")

        self.select_tides_button = Button(
            description="Select Tides",
            style=load_style,
            icon="fa-file-image-o",
        )
        self.select_tides_button.on_click(self.select_tides_button_clicked)

        self.tidally_correct_button = Button(
            description="Correct Tides",
            style=load_style,
            icon="fa-tint",
        )
        self.tidally_correct_button.on_click(self.tidally_correct_button_clicked)

        return VBox(
            [
                self.beach_slope_text,
                self.reference_elevation_text,
                self.select_tides_button,
                self.tidally_correct_button,
            ]
        )

    @tidal_correction_view.capture(clear_output=True)
    def tidally_correct_button_clicked(self, button):
        # user must have selected imagery first
        # must select a directory of model outputs
        if self.shoreline_session_directory == "":
            self.launch_error_box(
                "Cannot correct tides",
                "Must click select session first",
                position=3,
            )
            return
        if self.tides_file == "":
            self.launch_error_box(
                "Cannot correct tides",
                "Must enter a select a tide file first",
                position=3,
            )
            return
        # get session directory location
        # print("Correcting tides... please wait")
        session_directory = self.shoreline_session_directory
        logger.info(f"session_directory: {session_directory}")
        # get roi_id
        # @todo find a better way to load the roi id
        config_json_location = common.find_file_recursively(
            session_directory, "config.json"
        )
        config = common.load_data_from_json(config_json_location)
        roi_id = config.get("roi_id", "")
        logger.info(f"roi_id: {roi_id}")

        beach_slope = self.beach_slope_text.value
        reference_elevation = self.reference_elevation_text.value
        # load in shoreline settings, session directory with model outputs, and a new session name to store extracted shorelines
        zoo_model.compute_tidal_corrections(
            roi_id,
            session_directory,
            session_directory,
            self.tides_file,
            beach_slope,
            reference_elevation,
        )

    @tidal_correction_view.capture(clear_output=True)
    def select_tides_button_clicked(self, button):
        # Prompt the user to select a directory of images
        file_chooser = common.create_file_chooser(
            self.load_tide_callback,
            title="Select csv file",
            filter_pattern="*csv",
            starting_directory="sessions",
        )
        # clear row and close all widgets in  self.tidal_correct_file_row before adding new file_chooser
        common.clear_row(self.tidal_correct_file_row)
        # add instance of file_chooser to  self.tidal_correct_file_row
        self.tidal_correct_file_row.children = [file_chooser]

    @tidal_correction_view.capture(clear_output=True)
    def load_tide_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            self.tides_file = os.path.abspath(filechooser.selected)

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
        load_style = dict(button_color="#69add1", description_width="initial")
        action_style = dict(button_color="#ae3cf0")

        self.run_model_button = Button(
            description="Run Model",
            style=action_style,
            icon="fa-bolt",
        )
        self.run_model_button.on_click(self.run_model_button_clicked)

        self.extract_shorelines_button = Button(
            description="Extract Shorelines", style=action_style
        )
        self.extract_shorelines_button.on_click(self.extract_shorelines_button_clicked)

        self.use_select_images_button = Button(
            description="Select Images",
            style=load_style,
            icon="fa-file-image-o",
        )
        self.use_select_images_button.on_click(self.use_select_images_button_clicked)

        self.select_model_session_button = Button(
            description="Select Model Session",
            style=load_style,
        )
        self.select_model_session_button.on_click(self.select_model_session_clicked)

        self.select_extracted_shorelines_session_button = Button(
            description="Select Session",
            style=load_style,
        )
        self.select_extracted_shorelines_session_button.on_click(
            self.select_extracted_shorelines_button_clicked
        )

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.line_widget = HTML(
            value="____________________________________________________"
        )
        self.run_model_instr = HTML(
            value="<h2>Run a Model</h2>\
            <b>1. Session Name:</b> Enter a name.This creates a folder in the 'sessions' directory for model outputs.\
            <br><b>2. Select Images:</b> Choose an ROI directory with downloaded imagery from the 'data' directory.\
            <br><b>3. Run Model:</b> Click this button to apply the model on imagery. Outputs are saved under your session name in the 'sessions' directory.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.extract_shorelines_instr = HTML(
            value="<h2>Extract Shorelines</h2>\
            Make sure to run the model before extracting shorelines.<br> \
            <b>1. Name Your Session:</b> Enter a name for your shoreline extraction session this will create a new folder in the 'sessions' directory.<br>\
            <b>2. Upload a File (Optional): </b> Upload a transects or a reference shoreline geojson file to extract shorelines with.<br>\
            <b>3. Choose Model Session</b> Select a directory from the 'sessions' directory which contains model outputs.<br>\
            <b>4. Click Extract Shorelines</b> Your extracted shorelines will be saved in the 'sessions' directory under your named session.<br><br>\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.tidally_correct_instr = HTML(
            value="<h2>Tidally Correct</h2>\
            Ensure shorelines are extracted prior to tidal correction. Not all imagery will contain extractable shorelines, thus, tidal correction may not be possible.\
            <br><b>1. Select a Session </b> Choose a session from the 'sessions' directory containing extracted shorelines.\
            <br><b>2. Select Tides Button</b> Pick a CSV file with tide levels and dates for the ROI.\
            <br><b>3. Correct Tides Button</b> Click to save tidally corrected CSV files in the selected session directory.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

    def handle_model_implementation(self, change):
        self.model_dict["implementation"] = change["new"]

    def handle_model_type(self, change):
        # 2 class model has not been selected disable otsu threhold
        self.model_dict["model_type"] = change["new"]
        logger.info(f"self.model_dict['model_type']: {self.model_dict['model_type']}")
        if "2class" not in change["new"]:
            if self.otsu_radio.value == "Enabled":
                self.model_dict["otsu"] = False
                self.otsu_radio.value = "Disabled"
            self.otsu_radio.disabled = True
        # 2 class model was selected enable otsu threhold radio button
        if "2class" in change["new"]:
            self.otsu_radio.disabled = False
        logger.info(f"change: {change}")

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
        # if change["new"] == "RGB+MNDWI+NDWI":
        #     self.model_dropdown.options = self.five_band_models

    @run_model_view.capture(clear_output=True)
    def run_model_button_clicked(self, button):
        # user must have selected imagery first
        if self.model_dict["sample_direc"] is None:
            self.launch_error_box(
                "Cannot Run Model",
                "You must click 'Select Images' first",
                position=1,
            )
            return
        # user must have selected imagery first
        session_name = self.get_session_name()
        if session_name == "":
            self.launch_error_box(
                "Cannot Run Model",
                "Must enter a session name first",
                position=1,
            )
            return
        print("Running the model. Please wait.")
        zoo_model_instance = self.get_model_instance()
        img_type = self.model_input_dropdown.value
        self.model_dict["model_type"] = self.model_dropdown.value
        self.model_dict["implementation"] = self.model_implementation.value
        # get percent no data from settings
        settings = self.settings_dashboard.get_settings()
        percent_no_data = settings.get("percent_no_data", 50.0)

        if self.otsu_radio.value == "Enabled":
            self.model_dict["otsu"] = True
        if self.otsu_radio.value == "Disabled":
            self.model_dict["otsu"] = False
        if self.otsu_radio.value == "Enabled":
            self.model_dict["tta"] = True
        if self.otsu_radio.value == "Disabled":
            self.model_dict["tta"] = False
        zoo_model_instance.run_model(
            img_type,
            self.model_dict["implementation"],
            session_name,
            self.model_dict["sample_direc"],
            model_name=self.model_dict["model_type"],
            use_GPU="0",
            use_otsu=self.model_dict["otsu"],
            use_tta=self.model_dict["tta"],
            percent_no_data=percent_no_data,
        )

    @extract_shorelines_view.capture(clear_output=True)
    def extract_shorelines_button_clicked(self, button):
        # user must have selected imagery first
        session_name = self.get_shoreline_session_name()
        if session_name == "":
            self.launch_error_box(
                "Cannot Extract Shorelines",
                "Must enter a session name first",
                position=2,
            )
            return
        # must select a directory of model outputs
        if self.model_session_directory == "":
            self.launch_error_box(
                "Cannot Extract Shorelines",
                "Must click select model session first",
                position=2,
            )
            return

        print("Extracting shorelines. Please wait.")
        transects_path = self.fileuploader.files_dict.get("transects", "")
        shoreline_path = self.fileuploader.files_dict.get("shorelines", "")

        shoreline_settings = self.settings_dashboard.get_settings()
        # get session directory location
        session_directory = self.model_session_directory
        zoo_model_instance = self.get_model_instance()
        # load in shoreline settings, session directory with model outputs, and a new session name to store extracted shorelines
        zoo_model_instance.extract_shorelines_with_unet(
            shoreline_settings,
            session_directory,
            session_name,
            shoreline_path,
            transects_path,
        )

    @run_model_view.capture(clear_output=True)
    def select_RGB_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            sample_direc = os.path.abspath(filechooser.selected)
            print(f"The images in the folder will be segmented :\n{sample_direc} ")
            jpgs = glob.glob1(sample_direc + os.sep, "*jpg")
            if jpgs == []:
                self.launch_error_box(
                    "File Not Found",
                    "The directory contains no jpgs! Please select a directory with jpgs.",
                    position=1,
                )
            elif jpgs != []:
                self.model_dict["sample_direc"] = sample_direc

    @run_model_view.capture(clear_output=True)
    def use_select_images_button_clicked(self, button):
        # Prompt the user to select a directory of images
        file_chooser = common.create_dir_chooser(
            self.select_RGB_callback, title="Select directory of images"
        )
        # clear row and close all widgets in self.file_row before adding new file_chooser
        common.clear_row(self.file_row)
        # add instance of file_chooser to self.file_row
        self.file_row.children = [file_chooser]

    @extract_shorelines_view.capture(clear_output=True)
    def selected_model_session_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            self.model_session_directory = os.path.abspath(filechooser.selected)

    @tidal_correction_view.capture(clear_output=True)
    def selected_shoreline_session_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            self.shoreline_session_directory = os.path.abspath(filechooser.selected)

    @extract_shorelines_view.capture(clear_output=True)
    def select_model_session_clicked(self, button):
        # Prompt the user to select a directory of model outputs
        file_chooser = common.create_dir_chooser(
            self.selected_model_session_callback,
            title="Select model outputs",
            starting_directory="sessions",
        )
        # clear row and close all widgets in self.extracted_shoreline_file_row before adding new file_chooser
        common.clear_row(self.extracted_shoreline_file_row)
        # add instance of file_chooser to self.extracted_shoreline_file_row
        self.extracted_shoreline_file_row.children = [file_chooser]

    @tidal_correction_view.capture(clear_output=True)
    def select_extracted_shorelines_button_clicked(self, button):
        # Prompt the user to select a directory of extracted shorelines session
        file_chooser = common.create_dir_chooser(
            self.selected_shoreline_session_callback,
            title="Select extracted shorelines session",
            starting_directory="sessions",
        )
        # clear row and close all widgets in self.tidal_correct_file_row before adding new file_chooser
        common.clear_row(self.tidal_correct_file_row)
        # add instance of file_chooser to self.tidal_correct_file_row
        self.tidal_correct_file_row.children = [file_chooser]

    def launch_error_box(self, title: str = None, msg: str = None, position: int = 1):
        # Show user error message
        warning_box = common.create_warning_box(
            title=title, msg=msg, instructions=None, msg_width="60%", box_width="30%"
        )
        # clear row and close all widgets before adding new warning_box
        common.clear_row(self.get_warning_box(position))
        # add instance of warning_box to warning_row
        self.get_warning_box(position).children = [warning_box]
