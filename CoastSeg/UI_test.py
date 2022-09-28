# standard python imports
import os
import json
#internal python imports
from CoastSeg import download_roi
# external python imports
import ipywidgets
import geopandas as gpd
from ipyleaflet import GeoJSON
from google.auth import exceptions as google_auth_exceptions
from tkinter import Tk,filedialog
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
from tkinter import messagebox


class UI:
    # all instances of UI will share the same debug_view
    # this means that UI and coastseg_map must have a 1:1 relationship
    # Output wdiget used to print messages and exceptions created by CoastSeg_Map
    debug_view = Output(layout={'border': '1px solid black'})
    # Output wdiget used to print messages and exceptions created by download progress
    download_view = Output(layout={'border': '1px solid black'})

    def __init__(self, coastseg_map):
         # save an instance of coastseg_map
        self.coastseg_map = coastseg_map
        # button styles
        self.remove_style=dict(button_color = 'red')
        self.load_style=dict(button_color = '#69add1')
        self.action_style=dict(button_color = '#ae3cf0')
        self.save_style=dict(button_color = '#50bf8f')
        self.clear_stlye = dict(button_color = '#a3adac')

        # Controls size of ROIs generated on map
        small_roi_size = 3500
        large_roi_size = 4000
        self.fishnet_sizes={'small':small_roi_size,'large':large_roi_size}
        # Declare widgets and on click callbacks
        self.load_bbox_button = Button(description="Load bbox from file",style=self.load_style)
        self.load_bbox_button.on_click(self.on_load_bbox_clicked)
        # load buttons
        self.transects_button = Button(description="Load Transects",style=self.load_style)
        self.transects_button.on_click(self.on_transects_button_clicked)
        self.shoreline_button = Button(description="Load Shoreline",style=self.load_style)
        self.shoreline_button.on_click(self.on_shoreline_button_clicked)
        self.load_rois_button = Button(description="Load rois from file",style=self.load_style)
        self.load_rois_button.on_click(self.on_load_rois_clicked)
        # Save and Generate buttons
        self.gen_button =Button(description="Generate ROI",style=self.action_style)
        self.gen_button.on_click(self.on_gen_button_clicked)
        self.save_button = Button(description="Save ROI",style=self.save_style)
        self.save_button.on_click(self.on_save_button_clicked)
        self.save_bbox_button = Button(description="Save Bbox",style=self.save_style)
        self.save_bbox_button.on_click(self.on_save_bbox_button_clicked)
        self.download_button = Button(description="Download ROIs",style=self.action_style)
        self.download_button.on_click(self.download_button_clicked)
        # # Remove buttons
        self.clear_debug_button = Button(description="Clear TextBox", style = self.clear_stlye)
        self.clear_debug_button.on_click(self.clear_debug_view)
        self.remove_all_button = Button(description="Remove all",style=self.remove_style)
        self.remove_all_button.on_click(self.remove_all_from_map)
        self.remove_transects_button = Button(description="Remove transects",style=self.remove_style)
        self.remove_transects_button.on_click(self.remove_transects)
        self.remove_bbox_button = Button(description="Remove bbox",style=self.remove_style)
        self.remove_bbox_button.on_click(self.remove_bbox_from_map)
        self.remove_coastline_button = Button(description="Remove coastline",style=self.remove_style)
        self.remove_coastline_button.on_click(self.remove_coastline_from_map)
        self.remove_rois_button = Button(description="Remove ROIs",style=self.remove_style)
        self.remove_rois_button.on_click(self.remove_all_rois_from_map)

        slider_style = {'description_width': 'initial'}
        self.small_fishnet_slider=ipywidgets.IntSlider(
            value=small_roi_size,
            min=0,
            max=10000,
            step=100,
            description='Small Fishnet:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style=slider_style
        )

        self.large_fishnet_slider=ipywidgets.IntSlider(
            value=large_roi_size,
            min=1000,
            max=10000,
            step=100,
            description='Large Fishnet:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style=slider_style
        )

        # widget handlers
        self.small_fishnet_slider.observe(self.handle_small_slider_change,'value')
        self.large_fishnet_slider.observe(self.handle_large_slider_change,'value')


    def handle_small_slider_change(self, change):
        self.fishnet_sizes['small']=change['new']
    def handle_large_slider_change(self, change):
        self.fishnet_sizes['large']=change['new']


    @debug_view.capture(clear_output=True)
    def on_gen_button_clicked(self, btn):
        if self.coastseg_map.shapes_list == [] :
            print("Draw a bounding box on the coast first, then click Generate ROI.")
        else:
            UI.debug_view.clear_output(wait=True)
            self.coastseg_map.m.default_style = {'cursor': 'wait'}
            print("Generating ROIs please wait.")
            # Generate ROIs along the coastline within the bounding box
            self.coastseg_map.generate_ROIS_fishnet(self.fishnet_sizes['large'],self.fishnet_sizes['small'])
            # Add the Clickable ROIs to the map
            self.coastseg_map.add_geojson_layer_to_map()
            print("ROIs generated. Please Select at least one ROI and click Save ROI.")
            self.coastseg_map.m.default_style = {'cursor': 'default'}

    @debug_view.capture(clear_output=True)
    def on_transects_button_clicked(self,btn):
        if self.coastseg_map.shapes_list == [] :
            print("Draw a bounding box on the coast first, then click Load Transects.")
        else:
            UI.debug_view.clear_output(wait=True)
            self.coastseg_map.m.default_style = {'cursor': 'wait'}
            print("Loading transects please wait.")
            # Add the transects to the map
            self.coastseg_map.load_transects_on_map()
            print("Transects Loaded.")
            self.coastseg_map.m.default_style = {'cursor': 'default'}

    @debug_view.capture(clear_output=True)
    def on_load_rois_clicked(self, button):
        # Prompt the user to select a directory of images
        self.root = Tk()
        self.root.withdraw()                                        # Hide the main window.
        self.root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the self.root to the top of all windows.
        self.root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(),
                                                    filetypes=[('geojson','*.geojson')],
                                                    title = "Select a geojson file containing rois")
        # Save the filename as an attribute of the button
        if self.root.filename:
            roi_file= self.root.filename
            rois = gpd.read_file(roi_file)
            # style fishnet and convert to dictionary to be added to map
            rois_dict = self.coastseg_map.style_rois(rois)
            # Save the styled fishnet to data for interactivity to be added later
            self.coastseg_map.data = rois_dict
            # Add the Clickable ROIs to the map
            self.coastseg_map.add_geojson_layer_to_map()
            print(f"Loaded the rois from the file :\n{roi_file} ")
        else:
            messagebox.showerror("ROI Selection Error", "You must select a valid geojson file first!")
        self.root.destroy()

    @debug_view.capture(clear_output=True)
    def on_shoreline_button_clicked(self,btn):
        if self.coastseg_map.shapes_list == [] :
            print("Draw a bounding box on the coast first, then click Load Transects.")
        else:
            self.coastseg_map.m.default_style = {'cursor': 'wait'}
            UI.debug_view.clear_output(wait=True)
            print("Loading shoreline please wait.")
            # Add the transects to the map
            self.coastseg_map.load_shoreline_on_map()
            print("Shoreline loaded.")
            self.coastseg_map.m.default_style = {'cursor': 'default'}

    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        if self.coastseg_map.selected_ROI:
            self.coastseg_map.m.default_style = {'cursor': 'wait'}
            UI.debug_view.append_stdout("Scroll down past map to see download progress.")
            try:
                self.download_button_clicked.disabled=True
                # download_roi.download_imagery(self.coastseg_map.selected_ROI,pre_process_settings,dates,sat_list,collection)
            except google_auth_exceptions.RefreshError as exception:
                print(exception)
                print("Please authenticate with Google using the cell above: \n  'Authenticate and Initialize with Google Earth Engine (GEE)'")
        else:
            UI.debug_view.append_stdout("No ROIs were selected. \nPlease select at least one ROI and click 'Save ROI' to save these ROI for download.")
        self.download_button_clicked.disabled=False
        self.coastseg_map.m.default_style = {'cursor': 'default'}

    @debug_view.capture(clear_output=True)
    def on_save_bbox_button_clicked(self, btn):
        if self.coastseg_map.shapes_list != [] :
            UI.debug_view.clear_output(wait=True)
            # Save selected bbox to a geojson file
            self.coastseg_map.save_bbox_to_file()
            UI.debug_view.clear_output(wait=True)
            print("BBox have been saved. Saved to bbox.geojson")
        else:
            print("Draw a bounding box on the coast first")

    @debug_view.capture(clear_output=True)
    def on_save_button_clicked(self,btn):
        if self.coastseg_map.selected_set:
            if len(self.coastseg_map.selected_set) == 0:
                print("Must select at least 1 ROI first before you can save ROIs.")
            else:
                UI.debug_view.clear_output(wait=True)
                self.coastseg_map.save_roi_fishnet("fishnet_rois.geojson")
                print("Saving ROIs")
                UI.debug_view.clear_output(wait=True)
                print("ROIs have been saved. Now click Download ROI to download the ROIs using CoastSat")
        else:
            self.root = Tk()
            self.root.withdraw()                                        # Hide the main window.
            self.root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the self.root to the top of all windows.
            messagebox.showerror("ROI Selection Error", "No ROIs were selected.")
            # print("No ROIs were selected.")
            self.root.destroy()

    @debug_view.capture(clear_output=True)
    def on_load_bbox_clicked(self, button):
        # Prompt the user to select a directory of images
        self.root = Tk()
        self.root.withdraw()                                        # Hide the main window.
        self.root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the self.root to the top of all windows.
        self.root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(),
                                                    filetypes=[('geojson','*.geojson')],
                                                    title = "Select a geojson file containing bbox")
        # Save the filename as an attribute of the button
        if self.root.filename:
            bbox_file= self.root.filename
            bbox_geodf = gpd.read_file(bbox_file)
            bbox_geojson = bbox_geodf.to_json()
            bbox_dict = json.loads(bbox_geojson)
            self.coastseg_map.shapes_list.append(bbox_dict['features'][0]['geometry'])
            bbox_layer = GeoJSON(
                data=bbox_dict,
                name="Bbox",
                style={
                    'color': '#75b671',
                    'fill_color': '#75b671',
                    'opacity': 1,
                    'fillOpacity': 0.2,
                    'weight': 4},
            )
            self.coastseg_map.m.add_layer(bbox_layer)
            print(f"Loaded the rois from the file :\n{bbox_file} ")
        else:
            messagebox.showerror("File Selection Error", "You must select a valid geojson file first!")
        self.root.destroy()

    @debug_view.capture(clear_output=True)
    def on_save_bbox_button_clicked(self, btn):
        if self.coastseg_map.shapes_list != [] :
            UI.debug_view.clear_output(wait=True)
            # Save selected bbox to a geojson file
            self.coastseg_map.save_bbox_to_file()
            UI.debug_view.clear_output(wait=True)
            print("BBox have been saved. Saved to bbox.geojson")
        else:
            print("Draw a bounding box on the coast first")


    @debug_view.capture(clear_output=True)
    def on_save_button_clicked(self, btn):
        if self.coastseg_map.selected_set:
            if len(self.coastseg_map.selected_set) == 0:
                print("Must select at least 1 ROI first before you can save ROIs.")
            else:
                UI.debug_view.clear_output(wait=True)
                self.coastseg_map.save_roi_fishnet("fishnet_rois.geojson")
                print("Saving ROIs")
                UI.debug_view.clear_output(wait=True)
                print("ROIs have been saved. Now click Download ROI to download the ROIs using CoastSat")
        else:
            print("No ROIs were selected.")


    def remove_all_from_map(self, btn):
        self.coastseg_map.remove_all()
    def remove_transects(self, btn):
        self.coastseg_map.remove_transects()
    def remove_bbox_from_map(self, btn):
        self.coastseg_map.remove_bbox()
    def remove_coastline_from_map(self, btn):
        self.coastseg_map.remove_shoreline()
    def remove_all_rois_from_map(self, btn):
        self.coastseg_map.remove_all_rois()
    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()
        UI.download_view.clear_output()