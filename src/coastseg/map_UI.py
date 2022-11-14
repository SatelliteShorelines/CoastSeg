# standard python imports
import os
import logging
import traceback
import sys

# internal python imports
from coastseg.tkinter_window_creator import Tkinter_Window_Creator
from coastseg import exceptions
from coastseg import exception_handler

# external python imports
import ipywidgets
from coastseg.tkinter_window_creator import Tkinter_Window_Creator
from IPython.display import display, clear_output
from google.auth import exceptions as google_auth_exceptions
from tkinter import filedialog
from tkinter import messagebox
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import DatePicker
from ipywidgets import HTML
from ipywidgets import RadioButtons
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
        # Declare widgets and on click callbacks
        self.load_gdf_button = Button(
            description="Load gdf from file", style=self.load_style
        )
        self.load_gdf_button.on_click(self.on_load_gdf_clicked)
        self.load_bbox_button = Button(
            description="Load bbox from file", style=self.load_style
        )
        self.load_bbox_button.on_click(self.on_load_bbox_clicked)
        # buttons to load configuration files
        self.load_configs_button = Button(
            description="Load Config", style=self.load_style
        )
        self.load_configs_button.on_click(self.on_load_configs_clicked)
        self.save_config_button = Button(
            description="Save Config", style=self.save_style
        )
        self.save_config_button.on_click(self.on_save_config_clicked)
        # load buttons
        self.transects_button = Button(
            description="Load Transects", style=self.load_style
        )
        self.transects_button.on_click(self.on_transects_button_clicked)
        self.shoreline_button = Button(
            description="Load Shoreline", style=self.load_style
        )
        self.shoreline_button.on_click(self.on_shoreline_button_clicked)
        self.load_rois_button = Button(
            description="Load rois from file", style=self.load_style
        )
        self.load_rois_button.on_click(self.on_load_rois_clicked)
        # Save buttons
        self.save_shoreline_button = Button(
            description="Save shorelines", style=self.save_style
        )
        self.save_shoreline_button.on_click(self.save_shoreline_button_clicked)
        self.save_transects_button = Button(
            description="Save transects", style=self.save_style
        )
        self.save_transects_button.on_click(self.save_transects_button_clicked)
        self.save_roi_button = Button(description="Save ROI", style=self.save_style)
        self.save_roi_button.on_click(self.save_roi_button_clicked)
        self.save_bbox_button = Button(description="Save box", style=self.save_style)
        self.save_bbox_button.on_click(self.on_save_bbox_button_clicked)
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
        self.remove_all_button = Button(
            description="Remove all", style=self.remove_style
        )
        self.remove_all_button.on_click(self.remove_all_from_map)
        self.remove_transects_button = Button(
            description="Remove transects", style=self.remove_style
        )
        self.remove_transects_button.on_click(self.remove_transects)
        self.remove_bbox_button = Button(
            description="Remove bbox", style=self.remove_style
        )
        self.remove_bbox_button.on_click(self.remove_bbox_from_map)
        self.remove_shoreline_button = Button(
            description="Remove shoreline", style=self.remove_style
        )
        self.remove_shoreline_button.on_click(self.remove_shoreline_from_map)
        self.remove_rois_button = Button(
            description="Remove ROIs", style=self.remove_style
        )
        self.remove_rois_button.on_click(self.remove_all_rois_from_map)

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

        self.instr_save_roi = HTML(
            value="<h2><b>Save Features</b></h2> \
                Use these buttons to save features on the map to a geojson file.\
                <br>These geojson files are saved to the CoastSeg directory.\
                </br><b>Save ROI</b>: Saves ROIs you selected to a file: 'rois.geojson'\
                </br><b>Save box</b>: Saves bounding box to a file: 'bbox.geojson'</li>\
                </br><b>Save shorelines</b>: Saves shorelines to a file: 'shoreline.geojson'</li>\
                </br><b>Save transects</b>: Saves selected ROI to a file: 'transects.geojson'</li>\
                ",
            layout=Layout(margin="0px 5px 0px 5px"),
        )  # top right bottom left

        self.instr_download_roi = HTML(
            value="<h2><b>Download ROIs</b></h2> \
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

        self.instr_load_btns = HTML(
            value="<h2><b>Load Features</b></h2>\
                You can upload ROIs or Bbox geojson file.\
                </br><b>Load BBox</b>: Load bounding box from file: 'bbox.geojson'</li>\
                </br><b>Load ROIs</b>: Load ROIs from file: 'rois.geojson'</li>\
                </br><b>Load gdf</b>: Load any geodataframe from a geojson file </li>\
                    ",
            layout=Layout(margin="0px 5px 0px 5px"),
        )  # top right bottom left

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
        save_vbox = VBox(
            [
                self.instr_save_roi,
                self.save_roi_button,
                self.save_bbox_button,
                self.save_shoreline_button,
                self.save_transects_button,
                self.instr_load_btns,
                self.load_rois_button,
                self.load_bbox_button,
                self.load_gdf_button,
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
            [self.instr_create_roi, slider_btn_box],
            layout=Layout(margin="0px 5px 5px 0px"),
        )

        load_buttons = HBox([self.transects_button, self.shoreline_button])
        erase_buttons = HBox(
            [
                self.remove_all_button,
                self.remove_transects_button,
                self.remove_bbox_button,
                self.remove_shoreline_button,
                self.remove_rois_button,
            ]
        )

        row_1 = HBox([roi_controls_box, save_vbox, download_vbox])
        row_2 = HBox([load_buttons])
        row_3 = HBox([erase_buttons])
        # in this row prints are rendered with UI.debug_view
        row_4 = HBox([self.clear_debug_button, UI.debug_view])
        row_5 = HBox([self.coastseg_map.map])
        row_6 = HBox([UI.download_view])

        return display(row_1, row_2, row_3, row_4, row_5, row_6)

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
    def on_transects_button_clicked(self, btn):

        UI.debug_view.clear_output(wait=True)
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        try:
            self.coastseg_map.load_transects_on_map()
        except Exception as error:
            exception_handler.handle_exception(error)
        self.coastseg_map.map.default_style = {"cursor": "default"}

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
    def on_shoreline_button_clicked(self, btn):
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        UI.debug_view.clear_output(wait=True)
        print("Loading shoreline please wait...")
        try:
            self.coastseg_map.load_shoreline_on_map()
        except Exception as error:
            exception_handler.handle_exception(error)
        print("Shoreline loaded.")
        self.coastseg_map.map.default_style = {"cursor": "default"}

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
    def save_transects_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(
                self.coastseg_map.transects, "transects"
            )
        except Exception as error:
            exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def save_shoreline_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(
                self.coastseg_map.shoreline, "shoreline"
            )
        except Exception as error:
            exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def on_save_cross_distances_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_transects_to_csv()
        except Exception as error:
            exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def on_save_bbox_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(
                self.coastseg_map.bbox, "bounding box"
            )
        except Exception as error:
            exception_handler.handle_exception(error)

    @debug_view.capture(clear_output=True)
    def on_load_gdf_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename = filedialog.askopenfilename(
                initialdir=os.getcwd(),
                filetypes=[("geojson", "*.geojson")],
                title="Select a geojson file",
            )
            # Save the filename as an attribute of the button
            if tk_root.filename:
                try:
                    self.coastseg_map.load_gdf_on_map("geodataframe", tk_root.filename)
                except Exception as error:
                    exception_handler.handle_exception(error)
            else:
                messagebox.showerror(
                    "File Selection Error",
                    "You must select a valid geojson file first!",
                )

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
    def on_load_bbox_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename = filedialog.askopenfilename(
                initialdir=os.getcwd(),
                filetypes=[("geojson", "*.geojson")],
                title="Select a geojson file containing bbox",
            )
            # Save the filename as an attribute of the button
            if tk_root.filename:
                try:
                    self.coastseg_map.load_bbox_on_map(tk_root.filename)
                except Exception as error:
                    exception_handler.handle_exception(error)
            else:
                messagebox.showerror(
                    "File Selection Error",
                    "You must select a valid geojson file first!",
                )

    @debug_view.capture(clear_output=True)
    def save_roi_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(self.coastseg_map.rois, "ROI")
        except Exception as error:
            exception_handler.handle_exception(error)

    def remove_all_from_map(self, btn):
        try:
            self.coastseg_map.remove_all()
        except Exception as error:
            exception_handler.handle_exception(error)

    def remove_transects(self, btn):
        try:
            self.coastseg_map.remove_transects()
        except Exception as error:
            exception_handler.handle_exception(error)

    def remove_bbox_from_map(self, btn):
        try:
            self.coastseg_map.remove_bbox()
        except Exception as error:
            exception_handler.handle_exception(error)

    def remove_shoreline_from_map(self, btn):
        try:
            self.coastseg_map.remove_shoreline()
        except Exception as error:
            exception_handler.handle_exception(error)

    def remove_all_rois_from_map(self, btn):
        try:
            self.coastseg_map.remove_all_rois()
        except Exception as error:
            exception_handler.handle_exception(error)

    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()
        UI.download_view.clear_output()
