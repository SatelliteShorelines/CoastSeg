# standard python imports
import os
import datetime
import logging

# internal python imports
from coastseg.tkinter_window_creator import Tkinter_Window_Creator
from coastseg import exception_handler
from coastseg import common

# external python imports
import ipywidgets
from coastseg.tkinter_window_creator import Tkinter_Window_Creator
from IPython.display import display, clear_output
from google.auth import exceptions as google_auth_exceptions
from tkinter import filedialog
from tkinter import messagebox
from ipywidgets import Accordion
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import DatePicker
from ipywidgets import HTML
from ipywidgets import RadioButtons
from ipywidgets import Text
from ipywidgets import SelectMultiple
from ipywidgets import Output
from ipywidgets import Checkbox


logger = logging.getLogger(__name__)


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
        # button styles
        self.remove_style = dict(button_color="red")
        self.load_style = dict(button_color="#69add1")
        self.action_style = dict(button_color="#ae3cf0")
        self.save_style = dict(button_color="#50bf8f")
        self.clear_stlye = dict(button_color="#a3adac")

        # Controls size of ROIs generated on map
        small_roi_size = 3500
        large_roi_size = 4000
        self.fishnet_sizes = {"small": small_roi_size, "large": large_roi_size}

        # declare settings widgets
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

        # satellite selection widgets
        satellite_instr = HTML(
            value="<b>Pick multiple satellites:</b>\
                <br> - Pick multiple satellites by holding the control key> \
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

        self.settings_button = Button(
            description="Save Settings", style=self.action_style
        )
        self.settings_button.on_click(self.save_settings_clicked)

        self.output_epsg_text = Text(value="4326", description="Output epsg:")

        # create settings accordion
        settings_vbox = VBox(
            [dates_vbox, satellite_vbox, self.output_epsg_text, self.settings_button]
        )
        self.settings_accordion = Accordion(children=[settings_vbox])
        self.settings_accordion.set_title(0, "Settings")

        # buttons to load configuration files
        self.load_configs_button = Button(
            description="Load Config", style=self.load_style
        )
        self.load_configs_button.on_click(self.on_load_configs_clicked)
        self.save_config_button = Button(
            description="Save Config", style=self.save_style
        )
        self.save_config_button.on_click(self.on_save_config_clicked)

        self.load_file_instr = HTML(
            value="<h2>Load Feature from File</h2>\
                 Load a feature onto map from geojson file.\
                ",
            layout=Layout(padding="0px"),
        )

        self.load_file_radio = RadioButtons(
            options=[
                "Shoreline",
                "Transects",
                "Bbox",
            ],
            value="Shoreline",
            description="",
            disabled=False,
        )
        self.load_file_button = Button(
            description=f"Load {self.load_file_radio.value} file", style=self.load_style
        )
        self.load_file_button.on_click(self.load_feature_from_file)

        def change_load_file_btn_name(change):
            self.load_file_button.description = f"Load {str(change['new'])} file"

        self.load_file_radio.observe(change_load_file_btn_name, "value")

        # Generate buttons
        self.gen_button = Button(description="Generate ROI", style=self.action_style)
        self.gen_button.on_click(self.on_gen_button_clicked)
        self.download_button = Button(
            description="Download Imagery", style=self.action_style
        )
        self.download_button.on_click(self.download_button_clicked)
        self.extract_shorelines_button = Button(
            description="Extract Shorelines", style=self.action_style
        )
        self.extract_shorelines_button.on_click(self.extract_shorelines_button_clicked)
        self.compute_transect_button = Button(
            description="Compute Transects", style=self.action_style
        )
        self.compute_transect_button.on_click(self.compute_transect_button_clicked)
        self.save_transect_csv_button = Button(
            description="Save Transects CSV", style=self.action_style
        )
        self.save_transect_csv_button.on_click(
            self.on_save_cross_distances_button_clicked
        )
        # Remove buttons
        self.clear_debug_button = Button(
            description="Clear TextBox", style=self.clear_stlye
        )
        self.clear_debug_button.on_click(self.clear_debug_view)

        # create the HTML widgets containing the instructions
        self._create_HTML_widgets()
        # define slider widgets that control ROI size
        slider_style = {"description_width": "initial"}
        self.small_fishnet_slider = ipywidgets.IntSlider(
            value=small_roi_size,
            min=0,
            max=10000,
            step=100,
            description="Small Grid:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=slider_style,
        )

        self.large_fishnet_slider = ipywidgets.IntSlider(
            value=large_roi_size,
            min=1000,
            max=10000,
            step=100,
            description="Large Grid:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=slider_style,
        )

        # widget handlers
        self.small_fishnet_slider.observe(self.handle_small_slider_change, "value")
        self.large_fishnet_slider.observe(self.handle_large_slider_change, "value")

    def save_to_file_buttons(self):
        # save to file buttons
        save_instr = HTML(
            value="<h2>Save to file</h2>\
                Save feature on the map to a geojson file.\
                <br>Geojson file will be saved to CoastSeg directory.\
                </br><b>Example</b>: Saves bounding box to bbox.geojson\
            ",
            layout=Layout(padding="0px"),
        )

        self.save_radio = RadioButtons(
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
            description=f"Save {self.save_radio.value} to file", style=self.save_style
        )
        self.save_button.on_click(self.save_to_file_btn_clicked)

        def save_radio_changed(change):
            self.save_button.description = f"Save {str(change['new'])} to file"

        self.save_radio.observe(save_radio_changed, "value")
        save_vbox = VBox([save_instr, self.save_radio, self.save_button])
        return save_vbox

    def load_feature_on_map_buttons(self):
        load_instr = HTML(
            value="<h2>Load Feature into Bounding Box</h2>\
                </br> Loads shoreline or transects into bounding box on map.\
                </br>If no transects or shorelines exist, then\
               </br> draw bounding box somewhere else\
                ",
            layout=Layout(padding="0px"),
        )
        self.load_radio = RadioButtons(
            options=["Shoreline", "Transects"],
            value="Shoreline",
            description="",
            disabled=False,
        )
        self.load_button = Button(
            description=f"Load {self.load_radio.value}", style=self.load_style
        )
        self.load_button.on_click(self.load_button_clicked)

        def handle_load_radio_change(change):
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

        self.remove_radio = RadioButtons(
            options=["Shoreline", "Transects", "Bbox", "ROIs"],
            value="Shoreline",
            description="",
            disabled=False,
        )
        self.remove_button = Button(
            description=f"Remove {self.remove_radio.value}", style=self.remove_style
        )

        def handle_remove_radio_change(change):
            self.remove_button.description = f"Remove {str(change['new'])}"

        self.remove_button.on_click(self.remove_feature_from_map)
        self.remove_radio.observe(handle_remove_radio_change, "value")
        # define remove all button
        self.remove_all_button = Button(
            description="Remove all", style=self.remove_style
        )
        self.remove_all_button.on_click(self.remove_all_from_map)

        remove_buttons = VBox(
            [
                remove_instr,
                self.remove_radio,
                self.remove_button,
                self.remove_all_button,
            ]
        )
        return remove_buttons

    def get_settings_html(
        self,
        settings: dict,
    ):
        # Modifies html of accordion when transect is hovered over
        default = "unknown"
        keys = [
            "cloud_thresh",
            "dist_clouds",
            "output_epsg",
            "check_detection",
            "adjust_detection",
            "save_figure",
            "min_beach_area",
            "buffer_size",
            "min_length_sl",
            "cloud_mask_issue",
            "sand_color",
            "pan_off",
            "max_dist_ref",
            "along_dist",
            "sat_list",
            "landsat_collection",
            "dates",
        ]
        # returns a dict with keys in keys and if a key does not exist in feature its value is default str
        values = common.get_default_dict(default=default, keys=keys, fill_dict=settings)
        return """ 
        <h2>Settings</h2>
        <p>sat_list: {}</p>
        <p>dates: {}</p>
        <p>landsat_collection: {}</p>
        <p>cloud_thresh: {}</p>
        <p>dist_clouds: {}</p>
        <p>output_epsg: {}</p>
        <p>save_figure: {}</p>
        <p>min_beach_area: {}</p>
        <p>buffer_size: {}</p>
        <p>min_length_sl: {}</p>
        <p>cloud_mask_issue: {}</p>
        <p>sand_color: {}</p>
        <p>pan_off: {}</p>
        <p>max_dist_ref: {}</p>
        <p>along_dist: {}</p>
        """.format(
            values["sat_list"],
            values["dates"],
            values["landsat_collection"],
            values["cloud_thresh"],
            values["dist_clouds"],
            values["output_epsg"],
            values["save_figure"],
            values["min_beach_area"],
            values["buffer_size"],
            values["min_length_sl"],
            values["cloud_mask_issue"],
            values["sand_color"],
            values["pan_off"],
            values["max_dist_ref"],
            values["along_dist"],
        )

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.instr_create_roi = HTML(
            value="<h2><b>Generate ROIs</b></h2> \
                Use the two sliders to control the size of the ROIs generated.\
                </br><b>No Overlap</b>: Set small slider to 0 and large slider to ROI size.</li>\
                </br><b>Overlap</b>: Set small slider to a value and large slider to ROI size.</li>\
                </br><b>ROI units</b>: meters squared.</li>\
                </br><h3><b><u>How it Works</u></b></br></h3> \
                <li>Two grids composed of ROIs (squares) and created within\
                </br>the bounding box.\
                <li>Each ROI is created along the shoreline.\
                <li>If there is no shoreline then ROIs cannot be created.\
                <li>The slider controls the size of the individual ROIs created.\
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
            value="<h2><b>Load and Save Config Files</b></h2>\
                <b>Load Config</b>: Load rois, shorelines, transects and bounding box from file: \
                </br>'config_gdf.geojson'</li>\
                <li>Make sure 'config.json' is in the same directory as 'config_gdf.geojson'.</li>\
                <b>Save Config</b>: Saves rois, shorelines, transects and bounding box to file:\
                </br>'config_gdf.geojson'</li>\
                <li>If the ROIs have not been downloaded the config file is in main CoastSeg directory in file:\
                </br>'config_gdf.geojson'</li>\
                <li>If the ROIs have been downloaded the config file is in each ROI's folder in file:\
                </br>'config_gdf.geojson'</li>\
                <li>The 'config.json' will be saved in the same directory as the 'config_gdf.geojson'.</li>\
                <li>The 'config.json' contains the data for the ROI and map settings.</li>\
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
            ]
        )
        config_vbox = VBox(
            [self.instr_config_btns, self.load_configs_button, self.save_config_button]
        )
        download_vbox = VBox(
            [
                self.instr_download_roi,
                self.download_button,
                self.extract_shorelines_button,
                self.compute_transect_button,
                self.save_transect_csv_button,
                config_vbox,
            ]
        )

        slider_v_box = VBox([self.small_fishnet_slider, self.large_fishnet_slider])
        slider_btn_box = VBox([slider_v_box, self.gen_button])
        roi_controls_box = VBox(
            [self.instr_create_roi, slider_btn_box, load_buttons],
            layout=Layout(margin="0px 5px 5px 0px"),
        )

        self.settings_html = HTML()
        self.settings_html.value = self.get_settings_html(self.coastseg_map.settings)

        html_settings_accordion = Accordion(children=[self.settings_html])
        html_settings_accordion.set_title(0, "View Settings")

        row_0 = HBox([self.settings_accordion, html_settings_accordion])
        row_1 = HBox([roi_controls_box, save_vbox, download_vbox])
        # in this row prints are rendered with UI.debug_view
        row_3 = HBox([self.clear_debug_button, UI.debug_view])
        row_4 = HBox([self.coastseg_map.map])
        row_5 = HBox([UI.download_view])

        return display(
            row_0,
            row_1,
            row_3,
            row_4,
            row_5,
        )

    def handle_small_slider_change(self, change):
        self.fishnet_sizes["small"] = change["new"]

    def handle_large_slider_change(self, change):
        self.fishnet_sizes["large"] = change["new"]

    @debug_view.capture(clear_output=True)
    def on_gen_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        print("Generating ROIs please wait.")
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        # Generate ROIs along the coastline within the bounding box
        try:
            self.coastseg_map.load_rois_on_map(
                self.fishnet_sizes["large"], self.fishnet_sizes["small"]
            )
        except Exception as error:
            exception_handler.handle_exception(error)
        print("ROIs generated. Please Select at least one ROI and click Save ROI.")
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @debug_view.capture(clear_output=True)
    def load_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        try:
            if "shoreline" in btn.description.lower():
                print("Finding Shoreline")
                self.coastseg_map.load_feature_on_map("shoreline")
            if "transects" in btn.description.lower():
                print("Finding 'Transects'")
                self.coastseg_map.load_feature_on_map("transects")
        except Exception as error:
            exception_handler.handle_exception(error)
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @debug_view.capture(clear_output=True)
    def save_settings_clicked(self, btn):
        if self.satellite_selection.value:
            sat_list = list(self.satellite_selection.value)
            # Save dates selected by user
            dates = [str(self.start_date.value), str(self.end_date.value)]
            settings = {"output_epsg": int(self.output_epsg_text.value)}
            try:
                self.coastseg_map.save_settings(
                    sat_list=sat_list, dates=dates, **settings
                )
                self.settings_html.value = self.get_settings_html(
                    self.coastseg_map.settings
                )
            except Exception as error:
                exception_handler.handle_exception(error)
        elif not self.satellite_selection.value:
            try:
                raise Exception("Must select at least one satellite first")
            except Exception as error:
                exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def on_load_rois_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename = filedialog.askopenfilename(
                initialdir=os.getcwd(),
                filetypes=[("geojson", "*.geojson")],
                title="Select a geojson file containing rois",
            )
            # Save the filename as an attribute of the button
            if tk_root.filename:
                try:
                    self.coastseg_map.load_rois_on_map(file=tk_root.filename)
                except Exception as error:
                    exception_handler.handle_exception(error)
            else:
                messagebox.showerror(
                    "ROI Selection Error", "You must select a valid geojson file first!"
                )

    @debug_view.capture(clear_output=True)
    def extract_shorelines_button_clicked(self, btn):
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.extract_shorelines_button.disabled = True
        try:
            self.coastseg_map.extract_all_shorelines()
        except Exception as error:
            exception_handler.handle_exception(error)
        self.extract_shorelines_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @debug_view.capture(clear_output=True)
    def compute_transect_button_clicked(self, btn):
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.compute_transect_button.disabled = True
        try:
            self.coastseg_map.compute_transects()
        except Exception as error:
            exception_handler.handle_exception(error)
        self.compute_transect_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        UI.debug_view.append_stdout("Scroll down past map to see download progress.")
        try:
            self.download_button.disabled = True
            try:
                self.coastseg_map.download_imagery()
            except Exception as error:
                exception_handler.handle_exception(error)
        except google_auth_exceptions.RefreshError as exception:
            print(exception)
            with Tkinter_Window_Creator():
                messagebox.showerror(
                    "Authentication Error",
                    "Please authenticate with Google using the cell above: \n  'Authenticate and Initialize with Google Earth Engine (GEE)'",
                )
        self.download_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @debug_view.capture(clear_output=True)
    def on_save_cross_distances_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_transects_to_csv()
        except Exception as error:
            exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def on_load_configs_clicked(self, button):
        # Prompt user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename = filedialog.askopenfilename(
                initialdir=os.getcwd(),
                filetypes=[("geojson", "*.geojson")],
                title="Select a geojson file",
            )
            # Save filename as an attribute of button
            if tk_root.filename:
                try:
                    self.coastseg_map.load_configs(tk_root.filename)
                except Exception as error:
                    exception_handler.handle_exception(error)
            else:
                messagebox.showerror(
                    "File Selection Error",
                    "You must select a valid geojson file first!",
                )

    @debug_view.capture(clear_output=True)
    def on_save_config_clicked(self, button):
        try:
            self.coastseg_map.save_config()
        except Exception as error:
            exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def load_feature_from_file(self, btn):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            file = filedialog.askopenfilename(
                initialdir=os.getcwd(),
                filetypes=[("geojson", "*.geojson")],
                title="Select a geojson file containing bbox",
            )
            if file:
                if "shoreline" in btn.description.lower():
                    print(f"Loading shoreline from file: {file}")
                    self.coastseg_map.load_feature_on_map("shoreline", file)
                if "transects" in btn.description.lower():
                    print(f"Loading transects from file: {file}")
                    self.coastseg_map.load_feature_on_map("transects", file)
                if "bbox" in btn.description.lower():
                    print(f"Loading bounding box from file: {file}")
                    self.coastseg_map.load_feature_on_map("bbox", file)
            else:
                messagebox.showerror(
                    "File Selection Error",
                    "You must select a valid geojson file first!",
                )

    @debug_view.capture(clear_output=True)
    def remove_feature_from_map(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            # Prompt the user to select a directory of images
            if "shoreline" in btn.description.lower():
                print(f"Removing shoreline")
                self.coastseg_map.remove_shoreline()
            if "transects" in btn.description.lower():
                print(f"Removing  transects")
                self.coastseg_map.remove_transects()
            if "bbox" in btn.description.lower():
                print(f"Removing bounding box")
                self.coastseg_map.remove_bbox()
            if "rois" in btn.description.lower():
                print(f"Removing ROIs")
                self.coastseg_map.remove_all_rois()
        except Exception as error:
            exception_handler.handle_exception(error)

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
            exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def remove_all_from_map(self, btn):
        try:
            self.coastseg_map.remove_all()
        except Exception as error:
            exception_handler.handle_exception(error)

    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()
        UI.download_view.clear_output()
