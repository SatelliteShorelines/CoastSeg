# standard python imports
import os
import datetime
import logging
from collections import defaultdict

# internal python imports
from coastseg import exception_handler
from coastseg import common
from coastseg.watchable_slider import Extracted_Shoreline_widget

# external python imports
import ipywidgets
from IPython.display import display
from ipyfilechooser import FileChooser

from google.auth import exceptions as google_auth_exceptions
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import DatePicker
from ipywidgets import HTML
from ipywidgets import BoundedFloatText
from ipywidgets import SelectMultiple
from ipywidgets import Output
from ipywidgets import Select
from ipywidgets import BoundedIntText

from ipywidgets import FloatText

logger = logging.getLogger(__name__)

# icons sourced from https://fontawesome.com/v4/icons/

def convert_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        logger.error(f"Invalid date: {date_str}. Expected format: 'YYYY-MM-DD'.{e}")
        raise ValueError(f"Invalid date: {date_str}. Expected format: 'YYYY-MM-DD'.{e}")

class UI:
    # all instances of UI will share the same debug_view
    # this means that UI and coastseg_map must have a 1:1 relationship
    # Output widget used to print messages and exceptions created by CoastSeg_Map
    debug_view = Output(layout={"border": "1px solid black"})
    # Output widget used to print messages and exceptions created by download progress
    download_view = Output(layout={"border": "1px solid black"})

    def __init__(self, coastseg_map):
        # save an instance of coastseg_map
        self.coastseg_map = coastseg_map

        self.session_name = ""
        self.session_directory = ""
        self.tides_file = ""


        # the widget will update whenever the value of the extracted_shoreline_layer or number_extracted_shorelines changes
        self.extract_shorelines_widget = Extracted_Shoreline_widget(self.coastseg_map)
        # have the slider watch the extracted_shoreline_layer, number_extracted_shorelines,roi_selected_to_extract_shoreline
        self.extract_shorelines_widget.watch(
            self.coastseg_map.extracted_shoreline_layer
        )
        self.extract_shorelines_widget.watch(
            self.coastseg_map.number_extracted_shorelines
        )
        self.extract_shorelines_widget.watch(
            self.coastseg_map.roi_ids_with_extracted_shorelines
        )
        self.extract_shorelines_widget.set_load_extracted_shorelines_button_on_click(
            self.coastseg_map.load_extracted_shorelines_to_map
        )

        # create button styles
        self.create_styles()

        # buttons to load configuration files
        self.load_session_button = Button(
            description="Load Session", icon="fa-files-o", style=self.load_style
        )
        self.load_session_button.on_click(self.on_load_session_clicked)

        self.load_settings_button = Button(
            description="Load settings", icon="fa-file-o", style=self.load_style
        )
        self.load_settings_button.on_click(self.on_load_settings_clicked)

        self.settings_button = Button(
            description="Save Settings", icon="fa-floppy-o", style=self.action_style
        )
        self.settings_button.on_click(self.save_settings_clicked)
        self.settings_btn_row = VBox([self.settings_button, self.load_settings_button])

        self.load_file_instr = HTML(
            value="<h2>Load Feature from File</h2>\
                 Load a feature onto map from geojson file.\
                ",
            layout=Layout(padding="0px"),
        )

        self.load_file_radio = ipywidgets.Dropdown(
            options=["Shoreline", "Transects", "Bbox", "ROIs"],
            value="Shoreline",
            description="",
            disabled=False,
        )
        self.load_file_button = Button(
            description=f"Load {self.load_file_radio.value} file",
            icon="fa-file-o",
            style=self.load_style,
        )
        self.load_file_button.on_click(self.load_feature_from_file)

        def change_load_file_btn_name(change: dict):
            self.load_file_button.description = f"Load {str(change['new'])} file"

        self.load_file_radio.observe(change_load_file_btn_name, "value")

        # Generate buttons
        self.gen_button = Button(
            description="Generate ROI", icon="fa-globe", style=self.action_style
        )
        self.gen_button.on_click(self.gen_roi_clicked)
        self.download_button = Button(
            description="Download Imagery", icon="fa-download", style=self.action_style
        )
        self.download_button.on_click(self.download_button_clicked)

        self.extract_shorelines_button = Button(
            description="Extract Shorelines", style=self.action_style
        )
        self.extract_shorelines_button.on_click(self.extract_shorelines_button_clicked)

        # Clear  textbox button
        self.clear_debug_button = Button(
            description="Clear TextBox", style=self.clear_stlye
        )
        self.clear_debug_button.on_click(self.clear_debug_view)
        # Clear download messages button
        self.clear_downloads_button = Button(
            description="Clear Downloads", style=self.clear_stlye
        )
        self.clear_downloads_button.on_click(self.clear_download_view)

        # create the HTML widgets containing the instructions
        self._create_HTML_widgets()
        self.roi_slider_instr = HTML(value="<b>Choose Area of ROIs</b>")
        # controls the ROI units displayed
        self.units_radio = ipywidgets.Dropdown(
            options=["m²", "km²"],
            value="km²",
            description="Select Units:",
            disabled=False,
        )
        # create two float text boxes that will control size of ROI created
        self.sm_area_textbox = BoundedFloatText(
            value=1,
            min=0,
            max=98,
            step=1,
            description="Small ROI Area(km²):",
            style={"description_width": "initial"},
            disabled=False,
        )
        self.lg_area_textbox = BoundedFloatText(
            value=3,
            min=0,
            max=98,
            step=1,
            description="Large ROI Area(km²):",
            style={"description_width": "initial"},
            disabled=False,
        )

        # called when unit radio button is clicked
        def units_radio_changed(change: dict):
            """
            Change the maximum area allowed and the description of the small and large ROI area
            textboxes when the units radio is changed. When the units for area is m² the max ROI area size
            is 980000000 and when the units for area is m² max ROI area size
            is 98.

            Parameters:
            change (dict): event dictionary fired by clicking the units_radio button
            """
            try:
                MAX_AREA = 980000000
                STEP = 1000
                if change["name"] == "value":
                    if change["new"] == "m²":
                        MAX_AREA = 980000000
                        STEP = 1000
                        # convert km²to m²
                        new_sm_val = self.sm_area_textbox.value * (10**6)
                        new_lg_val = self.lg_area_textbox.value * (10**6)
                    elif change["new"] == "km²":
                        MAX_AREA = 98
                        STEP = 1
                        # convert m²to km²
                        new_sm_val = self.sm_area_textbox.value / (10**6)
                        new_lg_val = self.lg_area_textbox.value / (10**6)
                    # update the textboxes according to selected units
                    self.sm_area_textbox.max = MAX_AREA
                    self.sm_area_textbox.step = STEP
                    self.lg_area_textbox.max = MAX_AREA
                    self.lg_area_textbox.step = STEP
                    self.sm_area_textbox.value = new_sm_val
                    self.lg_area_textbox.value = new_lg_val
                    self.sm_area_textbox.description = (
                        f"Small ROI Area({self.units_radio.value}):"
                    )
                    self.lg_area_textbox.description = (
                        f"Large ROI Area({self.units_radio.value}):"
                    )
            except Exception as e:
                print(e)

        # when units radio button is clicked updated units for area textboxes
        self.units_radio.observe(units_radio_changed)

    def create_styles(self):
        """
        Initializes the styles used for various buttons in the user interface.
        Returns:
            None.
        """
        self.remove_style = dict(button_color="red")
        self.load_style = dict(button_color="#69add1", description_width="initial")
        self.action_style = dict(button_color="#ae3cf0")
        self.save_style = dict(button_color="#50bf8f")
        self.clear_stlye = dict(button_color="#a3adac")

    def launch_error_box(self, title: str = None, msg: str = None):
        # Show user error message
        warning_box = common.create_warning_box(title=title, msg=msg)
        # clear row and close all widgets in row before adding new warning_box
        common.clear_row(self.error_row)
        # add instance of warning_box to self.error_row
        self.error_row.children = [warning_box]

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

    @debug_view.capture(clear_output=True)
    def tidally_correct_button_clicked(self, button):
        if self.tides_file == "":
            self.launch_error_box(
                "Cannot correct tides",
                "Must enter a select a tide file first",
            )
            return

        print("Correcting tides... please wait")
        beach_slope = self.beach_slope_text.value
        reference_elevation = self.reference_elevation_text.value
        self.coastseg_map.compute_tidal_corrections(
            self.tides_file, beach_slope, reference_elevation
        )
        # load in shoreline settings, session directory with model outputs, and a new session name to store extracted shorelines

    @debug_view.capture(clear_output=True)
    def select_tides_button_clicked(self, button):
        # Prompt the user to select a directory of images
        file_chooser = common.create_file_chooser(
            self.load_tide_callback,
            title="Select csv file",
            filter_pattern="*csv",
            starting_directory="sessions",
        )
        # clear row and close all widgets in self.file_chooser_row before adding new file_chooser
        common.clear_row(self.file_chooser_row)
        # add instance of file_chooser to self.file_chooser_row
        self.file_chooser_row.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def load_tide_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            self.tides_file = os.path.abspath(filechooser.selected)

    def set_session_name(self, name: str):
        self.session_name = str(name).strip()

    def get_session_name(
        self,
    ):
        return self.session_name

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
            # create the session directory
            session_name = str(self.session_name_text.value).strip()
            session_path = os.path.join(os.getcwd(), "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(
                    f"Session {session_name} already exists at {new_session_path}\n Warning any existing files will be overwritten."
                )
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} was created at {new_session_path}")
                new_session_path = common.create_directory(session_path, session_name)
            self.coastseg_map.set_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def get_view_settings_vbox(self) -> VBox:
        # update settings button
        update_settings_btn = Button(
            description="Refresh Settings", icon="fa-refresh", style=self.action_style
        )
        update_settings_btn.on_click(self.update_settings_btn_clicked)
        self.settings_html = HTML()
        self.settings_html.value = self.get_settings_html(
            self.coastseg_map.get_settings()
        )
        view_settings_vbox = VBox([self.settings_html, update_settings_btn])
        return view_settings_vbox

    def get_settings_section(self):
        # declare settings widgets
        settings = {
            "dates_picker": self.get_dates_picker(),
            "satellite_radio": self.get_satellite_radio(),
            "sand_dropbox": self.get_sand_dropbox(),
            "min_length_sl_slider": self.get_min_length_sl_slider(),
            "beach_area_slider": self.get_beach_area_slider(),
            "shoreline_buffer_slider": self.get_shoreline_buffer_slider(),
            "cloud_slider": self.get_cloud_slider(),
            "cloud_threshold_slider": self.get_cloud_threshold_slider(),
            "along_dist": self.get_alongshore_distance_slider(),
            "min_points": self.get_min_points_text(),
            "max_std": self.get_max_std_text(),
            "max_range": self.get_max_range_text(),
            "min_chainage": self.get_min_chainage_text(),
            "multiple_inter": self.get_outliers_mode(),
            "prc_multiple": self.get_prc_multiple_text(),
        }

        # create settings vbox
        settings_vbox = VBox(
            [widget for widget_name, widget in settings.items()]
            + [self.settings_btn_row]
        )
        return settings_vbox

    def get_dates_picker(self):
        # Date Widgets
        self.start_date = DatePicker(
            description="Start Date",
            value=datetime.date(2018, 12, 1),
            disabled=False,
        )
        self.end_date = DatePicker(
            description="End Date",
            value=datetime.date(2019, 3, 1),  # 2019, 1, 1
            disabled=False,
        )
        date_instr = HTML(value="<b>Pick a date:</b>", layout=Layout(padding="10px"))
        dates_box = HBox([self.start_date, self.end_date])
        dates_vbox = VBox([date_instr, dates_box])
        return dates_vbox

    def get_cloud_threshold_slider(self):
        instr = HTML(value="<b>Maximum percetange of cloud pixels allowed</b>")
        self.cloud_threshold_slider = ipywidgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.01,
            description="Cloud Pixel %:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style={"description_width": "initial"},
        )
        return VBox([instr, self.cloud_threshold_slider])

    def get_sand_dropbox(self):
        sand_color_instr = HTML(
            value="<b>Sand color on beach for model to detect 'dark' (grey/black) 'bright' (white)</b>"
        )
        self.sand_dropdown = ipywidgets.Dropdown(
            options=["default", "latest", "dark", "bright"],
            value="default",
            description="Sand Color:",
            disabled=False,
        )
        return VBox([sand_color_instr, self.sand_dropdown])

    def get_alongshore_distance_slider(self):
        # returns slider to control beach area slider
        instr = HTML(
            value="<b>Along-shore distance over which to consider shoreline points to compute median intersection with transects</b>"
        )
        self.alongshore_distance_slider = ipywidgets.IntSlider(
            value=25,
            min=10,
            max=100,
            step=1,
            description="Alongshore Distance:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        return VBox([instr, self.alongshore_distance_slider])

    def get_cloud_slider(self):
        # returns slider to control beach area slider
        cloud_instr = HTML(
            value="<b>Allowed distance from extracted shoreline to detected clouds</b>\
        </br>- Any extracted shorelines within this distance to any clouds will be dropped"
        )

        self.cloud_slider = ipywidgets.IntSlider(
            value=300,
            min=100,
            max=1000,
            step=1,
            description="Cloud Distance (m):",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        return VBox([cloud_instr, self.cloud_slider])

    def get_shoreline_buffer_slider(self):
        # returns slider to control beach area slider
        shoreline_buffer_instr = HTML(
            value="<b>Buffer around reference shorelines in which shorelines can be extracted</b>"
        )

        self.shoreline_buffer_slider = ipywidgets.IntSlider(
            value=50,
            min=100,
            max=500,
            step=1,
            description="Reference Shoreline Buffer (m):",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        return VBox([shoreline_buffer_instr, self.shoreline_buffer_slider])

    def get_beach_area_slider(self):
        # returns slider to control beach area slider
        beach_area_instr = HTML(
            value="<b>Minimum area (sqm) for object to be labelled as beach</b>"
        )

        self.beach_area_slider = ipywidgets.IntSlider(
            value=4500,
            min=1000,
            max=10000,
            step=10,
            description="Beach Area (sqm):",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        return VBox([beach_area_instr, self.beach_area_slider])

    def get_min_chainage_text(self) -> VBox:
        # returns slider to control beach area slider
        label = HTML(
            value="<b>Max distance landward of the transect origin that an intersection is accepted, beyond this point a NaN is returned.</b>"
        )

        # min_chainage: (in metres) furthest distance landward of the transect origin that an intersection is accepted, beyond this point a NaN is returned.
        self.min_chainage_text = BoundedFloatText(
            value=-100.0,
            min=-500.0,
            max=-1.0,
            step=-1.0,
            description="Max Landward Distance",
            style={"description_width": "initial"},
            disabled=False,
        )
        return VBox([label, self.min_chainage_text])

    def get_prc_multiple_text(self) -> VBox:
        # returns slider to control beach area slider
        label = HTML(
            value="<b>Percentage of points whose std > max_std that will be set to 'max'.Only in 'auto' mode.</b>"
        )
        # percentage of points whose std > max_std that will be set to 'max'
        # percentage of data points where the std is larger than the user-defined max
        # 'prc_multiple': percentage to use in 'auto' mode to switch from 'nan' to 'max'
        self.prc_multiple_text = BoundedFloatText(
            value=0.1,
            min=0.0,
            max=1.0,
            step=0.01,
            description="% points' std > max_std:",
            style={"description_width": "initial"},
            disabled=False,
        )
        return VBox([label, self.prc_multiple_text])

    def get_max_range_text(self) -> VBox:
        # returns slider to control beach area slider
        label = HTML(
            value="<b>Max range for shoreline points within the alongshore range, if range is above this value a NaN is returned for this intersection</b>"
        )
        # max_range: (in metres) maximum RANGE for the shoreline points within the alongshore range, if RANGE is above this value a NaN is returned for this intersection.
        self.max_range_text = BoundedFloatText(
            value=30.0,
            min=1.0,
            max=100.0,
            step=1.0,
            description="Max Range",
            style={"description_width": "initial"},
            disabled=False,
        )
        return VBox([label, self.max_range_text])

    def get_outliers_mode(self) -> VBox:
        # returns slider to control beach area slider
        label = HTML(value="<b>How to deal with multiple shoreline intersections.</b>")
        # controls multiple_inter: ('auto','nan','max') defines how to deal with multiple shoreline intersections
        self.outliers_mode = Select(
            options=["auto", "nan", "max"],
            value="auto",
            description="Outliers Mode",
            style={"description_width": "initial"},
        )
        return VBox([label, self.outliers_mode])

    def get_max_std_text(self) -> VBox:
        # returns slider to control beach area slider
        label = HTML(
            value="<b>Maximum STD for the shoreline points within the alongshore range</b>"
        )

        # max_std: (in metres) maximum STD for the shoreline points within the alongshore range, if STD is above this value a NaN is returned for this intersection.
        self.max_std_text = BoundedFloatText(
            value=15.0,
            min=1.0,
            max=100.0,
            step=1.0,
            description="Max Std",
            style={"description_width": "initial"},
            disabled=False,
        )
        return VBox([label, self.max_std_text])

    def get_min_points_text(self) -> VBox:
        # returns slider to control beach area slider
        label = HTML(
            value="<b>Minimum number of shoreline points to calculate an intersection</b>"
        )

        # min_points: minimum number of shoreline points to calculate an intersection.
        self.min_points_text = BoundedIntText(
            value=3,
            min=1,
            max=100,
            step=1,
            description="Min Shoreline Points",
            style={"description_width": "initial"},
            disabled=False,
        )
        return VBox([label, self.min_points_text])

    def get_min_length_sl_slider(self):
        # returns slider to control beach area slider
        min_length_sl_instr = HTML(
            value="<b>Minimum shoreline perimeter that model will detect</b>"
        )

        self.min_length_sl_slider = ipywidgets.IntSlider(
            value=500,
            min=200,
            max=1000,
            step=1,
            description="Min shoreline length (m):",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        return VBox([min_length_sl_instr, self.min_length_sl_slider])

    def get_satellite_radio(self):
        # satellite selection widgets
        satellite_instr = HTML(
            value="<b>Pick multiple satellites:</b>\
                <br> - Pick multiple satellites by holding the control key \
                <br> - images after 2022/01/01 will be automatically downloaded from Collection 2 ",
            layout=Layout(padding="10px"),
        )

        self.satellite_selection = SelectMultiple(
            options=["L5", "L7", "L8", "L9", "S2"],
            value=["L8"],
            description="Satellites",
            disabled=False,
        )
        satellite_vbox = VBox([satellite_instr, self.satellite_selection])
        return satellite_vbox

    def save_to_file_buttons(self):
        # save to file buttons
        save_instr = HTML(
            value="<h2>Save to file</h2>\
                Save feature on the map to a geojson file.\
                <br>Geojson file will be saved to CoastSeg directory.\
            ",
            layout=Layout(padding="0px"),
        )

        self.save_radio = ipywidgets.Dropdown(
            options=[
                "Shoreline",
                "Transects",
                "Bbox",
                "ROIs",
            ],
            value="Shoreline",
            description="",
            disabled=False,
        )

        self.save_button = Button(
            description=f"Save {self.save_radio.value}",
            icon="fa-floppy-o",
            style=self.save_style,
        )
        self.save_button.on_click(self.save_to_file_btn_clicked)

        def save_radio_changed(change: dict):
            self.save_button.description = f"Save {str(change['new'])} to file"

        self.save_radio.observe(save_radio_changed, "value")
        save_vbox = VBox([save_instr, self.save_radio, self.save_button])
        return save_vbox

    def load_feature_on_map_buttons(self):
        load_instr = HTML(
            value="<h2>Load Feature into Bounding Box</h2>\
                Loads shoreline or transects into bounding box on map.\
                </br>If no transects or shorelines exist in this area, then\
               </br> draw bounding box somewhere else\
                ",
            layout=Layout(padding="0px"),
        )
        self.load_radio = ipywidgets.Dropdown(
            options=["Shoreline", "Transects"],
            value="Transects",
            description="",
            disabled=False,
        )
        self.load_button = Button(
            description=f"Load {self.load_radio.value}",
            icon="fa-file-o",
            style=self.load_style,
        )
        self.load_button.on_click(self.load_button_clicked)

        def handle_load_radio_change(change: dict):
            self.load_button.description = f"Load {str(change['new'])}"

        self.load_radio.observe(handle_load_radio_change, "value")
        load_buttons = VBox([load_instr, self.load_radio, self.load_button])
        return load_buttons

    def remove_buttons(self):
        # define remove feature radio box button
        remove_instr = HTML(
            value="<h2>Remove Feature from Map</h2>",
            layout=Layout(padding="0px"),
        )

        self.feature_dropdown = ipywidgets.Dropdown(
            options=["Shoreline", "Transects", "Bbox", "ROIs", "Extracted Shorelines"],
            value="Shoreline",
            description="",
            disabled=False,
        )
        self.remove_button = Button(
            description=f"Remove {self.feature_dropdown.value}",
            icon="fa-ban",
            style=self.remove_style,
        )

        def handle_remove_radio_change(change: dict):
            self.remove_button.description = f"Remove {str(change['new'])}"

        self.remove_button.on_click(self.remove_feature_from_map)
        self.feature_dropdown.observe(handle_remove_radio_change, "value")
        # define remove all button
        self.remove_all_button = Button(
            description="Remove all", icon="fa-trash-o", style=self.remove_style
        )
        self.remove_all_button.on_click(self.remove_all_from_map)

        remove_buttons = VBox(
            [
                remove_instr,
                self.feature_dropdown,
                self.remove_button,
                self.remove_all_button,
            ]
        )
        return remove_buttons

    def update_settings_selection(
        self,
        settings: dict,
    ):
        if "dates" in settings:
            start_date_str, end_date_str = settings["dates"]
            logger.info(f"start_date_str, end_date_str {start_date_str, end_date_str}")
            self.start_date.value = convert_date(start_date_str)
            self.end_date.value = convert_date(end_date_str)


        if "cloud_thresh" in settings:
            self.cloud_threshold_slider.value = settings.get("cloud_thresh", 0.5)

        if "sat_list" in settings:
            self.satellite_selection.value = settings.get("sat_list", ["L8"])

        if "sand_color" in settings:
            self.sand_dropdown.value = settings.get("sand_color", "default")

        if "min_length_sl" in settings:
            self.min_length_sl_slider.value = settings.get("min_length_sl", 1000)

        if "dist_clouds" in settings:
            self.cloud_slider.value = settings.get("dist_clouds", 300)

        if "min_beach_area" in settings:
            self.beach_area_slider.value = settings.get("min_beach_area", 100)

        if "max_dist_ref" in settings:
            self.shoreline_buffer_slider.value = settings.get("max_dist_ref", 25)

        if "along_dist" in settings:
            self.alongshore_distance_slider.value = settings.get("along_dist", 25)

        if "max_std" in settings:
            self.max_std_text.value = settings.get("max_std", 15)

        if "max_range" in settings:
            self.max_range_text.value = settings.get("max_range", 30)

        if "min_chainage" in settings:
            self.min_chainage_text.value = settings.get("min_chainage", -100)

        if "multiple_inter" in settings:
            self.outliers_mode.value = settings.get("multiple_inter", "auto")

        if "prc_multiple" in settings:
            self.prc_multiple_text.value = settings.get("prc_multiple", 0.1)

        if "min_points" in settings:
            self.min_points_text.value = settings.get("min_points", 3)

    def get_settings_html(self, settings: dict):
        """
        Generates HTML content displaying the settings.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            str: The HTML content representing the settings.

        """

        return f"""
        <h2>Settings</h2>
        <p>sat_list: {settings.get("sat_list", "unknown")}</p>
        <p>dates: {settings.get("dates", "unknown")}</p>
        <p>landsat_collection: {settings.get("landsat_collection", "unknown")}</p>
        <p>cloud_thresh: {settings.get("cloud_thresh", "unknown")}</p>
        <p>dist_clouds: {settings.get("dist_clouds", "unknown")}</p>
        <p>output_epsg: {settings.get("output_epsg", "unknown")}</p>
        <p>save_figure: {settings.get("save_figure", "unknown")}</p>
        <p>min_beach_area: {settings.get("min_beach_area", "unknown")}</p>
        <p>min_length_sl: {settings.get("min_length_sl", "unknown")}</p>
        <p>cloud_mask_issue: {settings.get("cloud_mask_issue", "unknown")}</p>
        <p>sand_color: {settings.get("sand_color", "unknown")}</p>
        <p>max_dist_ref: {settings.get("max_dist_ref", "unknown")}</p>
        <p>along_dist: {settings.get("along_dist", "unknown")}</p>
        <p>min_points: {settings.get("min_points", "unknown")}</p>
        <p>max_std: {settings.get("max_std", "unknown")}</p>
        <p>max_range: {settings.get("max_range", "unknown")}</p>
        <p>min_chainage: {settings.get("min_chainage", "unknown")}</p>
        <p>multiple_inter: {settings.get("multiple_inter", "unknown")}</p>
        <p>prc_multiple: {settings.get("prc_multiple", "unknown")}</p>
        """

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.instr_create_roi = HTML(
            value="<h2><b>Generate ROIs on Map</b></h2> \
                </br><b>No Overlap</b>: Set Small ROI Area to 0 and Large ROI Area to ROI area.</li>\
                </br><b>Overlap</b>: Set Small ROI Area to a value and Large ROI Area to ROI area.</li>\
                </br><h3><b><u>How ROIs are Made</u></b></br></h3> \
                <li>Two grids of ROIs (squares) are created within\
                </br>the bounding box along the shoreline.\
                <li>If no shoreline is within the bounding box then ROIs cannot be created.\
                ",
            layout=Layout(margin="0px 5px 0px 0px"),
        )

        self.instr_download_roi = HTML(
            value="<h2><b>Download Imagery</b></h2> \
                <li><b>You must click an ROI on the map before you can download ROIs</b> \
                <li>Scroll past the map to see the download progress \
                </br><h3><b><u>Where is my data?</u></b></br></h3> \
                <li>The data you downloaded will be in the 'data' folder in the main CoastSeg directory</li>\
                Each ROI you downloaded will have its own folder with the ROI's ID and\
                </br>the time it was downloaded in the folder name\
                </br><b>Example</b>: 'ID_1_datetime11-03-22__02_33_22'</li>\
                ",
            layout=Layout(margin="0px 0px 0px 5px"),
        )

        self.instr_config_btns = HTML(
            value="<h2><b>Load Sessions</b></h2>\
                <b>Load Session</b>: Load rois, shorelines, transects, and bounding box from session directory\
                </br><b>ROIs Not Downloaded:</b> config file will be saved to CoastSeg directory in file: 'config_gdf.geojson'\
                </br><b>Downloaded ROIs:</b>config file will be saved to each ROI's directory in file: 'config_gdf.geojson'\
                ",
            layout=Layout(margin="0px 5px 0px 5px"),
        )  # top right bottom left

    def create_dashboard(self):
        """creates a dashboard containing all the buttons, instructions and widgets organized together."""
        # Buttons to load shoreline or transects in bbox on map
        load_buttons = self.load_feature_on_map_buttons()
        remove_buttons = self.remove_buttons()
        save_to_file_buttons = self.save_to_file_buttons()

        load_file_vbox = VBox(
            [self.load_file_instr, self.load_file_radio, self.load_file_button]
        )
        save_vbox = VBox(
            [
                save_to_file_buttons,
                load_file_vbox,
                remove_buttons,
                self.extract_shorelines_widget,
            ]
        )
        config_vbox = VBox(
            [
                self.instr_config_btns,
                self.load_session_button,
            ]
        )
        download_vbox = VBox(
            [
                self.instr_download_roi,
                self.download_button,
                self.extract_shorelines_button,
                self.get_session_selection(),
                self.create_tidal_correction_widget(),
                config_vbox,
            ]
        )

        area_control_box = VBox(
            [
                self.roi_slider_instr,
                self.units_radio,
                self.sm_area_textbox,
                self.lg_area_textbox,
            ]
        )
        ROI_btns_box = VBox([area_control_box, self.gen_button])
        roi_controls_box = VBox(
            [self.instr_create_roi, ROI_btns_box, load_buttons],
            layout=Layout(margin="0px 5px 5px 0px"),
        )

        self.settings_row = HBox(
            [
                self.get_settings_section(),
                self.get_view_settings_vbox(),
            ]
        )
        row_1 = HBox([roi_controls_box, save_vbox, download_vbox])
        # in this row prints are rendered with UI.debug_view
        row_2 = HBox([self.clear_debug_button, UI.debug_view])
        self.error_row = HBox([])
        self.file_chooser_row = HBox([])
        map_row = HBox([self.coastseg_map.map])
        download_msgs_row = HBox([self.clear_downloads_button, UI.download_view])

        return display(
            self.settings_row,
            row_1,
            row_2,
            self.error_row,
            self.file_chooser_row,
            map_row,
            download_msgs_row,
        )

    @debug_view.capture(clear_output=True)
    def update_settings_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        # Update settings in view settings section
        try:
            self.settings_html.value = self.get_settings_html(
                self.coastseg_map.get_settings()
            )
        except Exception as error:
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def gen_roi_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        print("Generate ROIs button clicked")
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.gen_button.disabled = True
        # Generate ROIs along the coastline within the bounding box
        try:
            print("Generating ROIs please wait...")
            self.coastseg_map.load_feature_on_map(
                "rois",
                lg_area=self.lg_area_textbox.value,
                sm_area=self.sm_area_textbox.value,
                units=self.units_radio.value,
            )
        except Exception as error:
            print("ROIs could not be generated")
            self.coastseg_map.map.default_style = {"cursor": "default"}
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        else:
            self.coastseg_map.map.default_style = {"cursor": "default"}
            print("ROIs generated. Please Select at least one ROI on the map")
        self.coastseg_map.map.default_style = {"cursor": "default"}
        self.gen_button.disabled = False

    @debug_view.capture(clear_output=True)
    def load_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        try:
            if "shoreline" in btn.description.lower():
                print("Finding Shoreline")
                self.coastseg_map.load_feature_on_map("shoreline", zoom_to_bounds=True)
            if "transects" in btn.description.lower():
                print("Finding 'Transects'")
                self.coastseg_map.load_feature_on_map("transects", zoom_to_bounds=True)
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @debug_view.capture(clear_output=True)
    def save_settings_clicked(self, btn):
        if not self.satellite_selection.value:
            try:
                raise Exception("Must select at least one satellite first")
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        settings = {
            "sat_list": list(self.satellite_selection.value),
            "dates": [str(self.start_date.value), str(self.end_date.value)],
            "max_dist_ref": self.shoreline_buffer_slider.value,
            "along_dist": self.alongshore_distance_slider.value,
            "dist_clouds": self.cloud_slider.value,
            "min_beach_area": self.beach_area_slider.value,
            "min_length_sl": self.min_length_sl_slider.value,
            "sand_color": str(self.sand_dropdown.value),
            "cloud_thresh": self.cloud_threshold_slider.value,
            "min_points": self.min_points_text.value,
            "max_std": self.max_std_text.value,
            "max_range": self.max_range_text.value,
            "min_chainage": self.min_chainage_text.value,
            "multiple_inter": self.outliers_mode.value,
            "prc_multiple": self.prc_multiple_text.value,
        }
        try:
            self.coastseg_map.set_settings(**settings)
            self.settings_html.value = self.get_settings_html(
                self.coastseg_map.get_settings()
            )
            self.update_settings_selection(self.coastseg_map.get_settings())
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def extract_shorelines_button_clicked(self, btn):
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.extract_shorelines_button.disabled = True
        try:
            self.coastseg_map.extract_all_shorelines()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        self.extract_shorelines_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.download_button.disabled = True
        UI.debug_view.append_stdout("Scroll down past map to see download progress.")
        try:
            try:
                self.download_button.disabled = True
                self.coastseg_map.download_imagery()
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        except google_auth_exceptions.RefreshError as exception:
            print(exception)
            exception_handler.handle_exception(
                exception,
                self.coastseg_map.warning_box,
                title="Authentication Error",
                msg="Please authenticate with Google using the cell above: \n Authenticate and Initialize with Google Earth Engine (GEE)",
            )
        self.download_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    def clear_row(self, row: HBox):
        """close widgets in row/column and clear all children
        Args:
            row (HBox)(VBox): row or column
        """
        for index in range(len(row.children)):
            row.children[index].close()
        row.children = []

    @debug_view.capture(clear_output=True)
    def on_load_settings_clicked(self, button):
        self.settings_chooser_row = HBox([])

        # Prompt user to select a config geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    self.coastseg_map.load_settings(filechooser.selected)
                    self.settings_html.value = self.get_settings_html(
                        self.coastseg_map.get_settings()
                    )
                    self.update_settings_selection(self.coastseg_map.get_settings())
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        # create instance of chooser that calls load_callback
        file_chooser = common.create_file_chooser(
            load_callback, title="Select a settings json file", filter_pattern="*.json"
        )
        # clear row and close all widgets in row_4 before adding new file_chooser
        self.clear_row(self.settings_chooser_row)

        # add instance of file_chooser to row 4
        self.settings_chooser_row.children = [file_chooser]
        self.settings_btn_row.children = [
            self.settings_button,
            self.load_settings_button,
            self.settings_chooser_row,
        ]

    @debug_view.capture(clear_output=True)
    def on_load_session_clicked(self, button):
        # Prompt user to select a config geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    self.coastseg_map.map.default_style = {"cursor": "wait"}
                    self.coastseg_map.load_fresh_session(filechooser.selected)
                    logger.info(f"filechooser.selected: {filechooser.selected}")
                    session_name = os.path.basename(
                        os.path.abspath(filechooser.selected)
                    )
                    logger.info(f"session_name: {session_name}")
                    self.session_name_text.value = session_name
                    self.settings_html.value = self.get_settings_html(
                        self.coastseg_map.get_settings()
                    )
                    self.update_settings_selection(self.coastseg_map.get_settings())
                    self.coastseg_map.map.default_style = {"cursor": "default"}
            except Exception as error:
                # renders error message as a box on map
                self.coastseg_map.map.default_style = {"cursor": "default"}
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        # create instance of chooser that calls load_callback
        dir_chooser = common.create_dir_chooser(
            load_callback,
            title="Select Session Directory",
            starting_directory="sessions",
        )
        # clear row and close all widgets in row_4 before adding new file_chooser
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to row 4
        self.file_chooser_row.children = [dir_chooser]
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @debug_view.capture(clear_output=True)
    def load_feature_from_file(self, btn):
        # Prompt user to select a geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    if "shoreline" in btn.description.lower():
                        print(
                            f"Loading shoreline from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "shoreline",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
                    if "transects" in btn.description.lower():
                        print(
                            f"Loading transects from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "transects",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
                    if "bbox" in btn.description.lower():
                        print(
                            f"Loading bounding box from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "bbox",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
                    if "rois" in btn.description.lower():
                        print(
                            f"Loading ROIs from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "rois",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        # change title of filechooser based on feature selected
        title = "Select a geojson file"
        # create instance of chooser that calls load_callback
        if "shoreline" in btn.description.lower():
            title = "Select shoreline geojson file"
        if "transects" in btn.description.lower():
            title = "Select transects geojson file"
        if "bbox" in btn.description.lower():
            title = "Select bounding box geojson file"
        if "rois" in btn.description.lower():
            title = "Select ROI geojson file"
        # create instance of chooser that calls load_callback
        file_chooser = common.create_file_chooser(load_callback, title=title)
        # clear row and close all widgets in row_4 before adding new file_chooser
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to row 4
        self.file_chooser_row.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def remove_feature_from_map(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            # Prompt the user to select a directory of images
            if "extracted shorelines" in btn.description.lower():
                print(f"Removing extracted shoreline")
                self.coastseg_map.remove_extracted_shoreline_layers()
            elif "shoreline" in btn.description.lower():
                print(f"Removing shoreline")
                self.coastseg_map.remove_shoreline()
            elif "transects" in btn.description.lower():
                print(f"Removing  transects")
                self.coastseg_map.remove_transects()
            elif "bbox" in btn.description.lower():
                print(f"Removing bounding box")
                self.coastseg_map.remove_bbox()
            elif "rois" in btn.description.lower():
                print(f"Removing ROIs")
                self.coastseg_map.remove_all_rois()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def save_to_file_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            if "shoreline" in btn.description.lower():
                print(f"Saving shoreline to file")
                self.coastseg_map.save_feature_to_file(
                    self.coastseg_map.shoreline, "shoreline"
                )
            if "transects" in btn.description.lower():
                print(f"Saving transects to file")
                self.coastseg_map.save_feature_to_file(
                    self.coastseg_map.transects, "transects"
                )
            if "bbox" in btn.description.lower():
                print(f"Saving bounding box to file")
                self.coastseg_map.save_feature_to_file(
                    self.coastseg_map.bbox, "bounding box"
                )
            if "rois" in btn.description.lower():
                print(f"Saving ROIs to file")
                self.coastseg_map.save_feature_to_file(self.coastseg_map.rois, "ROI")
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def remove_all_from_map(self, btn):
        try:
            self.coastseg_map.remove_all()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()

    def clear_download_view(self, btn):
        UI.download_view.clear_output()
