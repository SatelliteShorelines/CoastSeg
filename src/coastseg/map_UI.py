# standard python imports
import os
import json
#internal python imports
from src.coastseg import download_roi
from src.coastseg.tkinter_window_creator import Tkinter_Window_Creator
from src.coastseg import exceptions
# external python imports
import ipywidgets
import geopandas as gpd
from ipyleaflet import GeoJSON
from IPython.display import display, clear_output
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
        self.remove_shoreline_button = Button(description="Remove shoreline",style=self.remove_style)
        self.remove_shoreline_button.on_click(self.remove_shoreline_from_map)
        self.remove_rois_button = Button(description="Remove ROIs",style=self.remove_style)
        self.remove_rois_button.on_click(self.remove_all_rois_from_map)
        # create the HTML widgets containing the instructions
        self._create_HTML_widgets()
        # define slider widgets that control ROI size
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

    def _create_HTML_widgets(self):
        """ create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.instr_create_roi=HTML(
            value="<h2><b>Generate ROIs</b></h2> \
                Use the two sliders to control the size of the ROIs generated.<br>\
                    <li> No Overlap: Set small slider to 0 and large slider to desired ROI size.</li>\
                    <li>Overlap: Set small slider to a value and large slider to desired ROI size.</li>\
                    <b>How it Works</b><br> \
                    Two fishnets (grids) are created out of ROIs (squares) and placed <br> \
                    within the bounding box.You control the size of the individual ROIs <br> \
                    that compose the small and large fishnets.<br>\
                    The ROIs are measured in meters squared.",layout=Layout(margin='0px 5px 0px 0px'))

        self.instr_save_roi=HTML(
            value="<h2><b>Save ROIs</b></h2> \
                You must click save ROI before you can download ROIs\
                <br> Use the save buttons to save the ROIs you selected <br>\
                <li> Save BBox: Save the bounding box to a geojson file called 'bbox.geojson'</li>\
                    <li> Save ROIs: Save the selected ROI to a geojson file called 'rois.geojson'</li>\
                    ",layout=Layout(margin='0px 5px 0px 5px')) #top right bottom left

        self.instr_download_roi=HTML(
            value="<h2><b>Download ROIs</b></h2> \
                <b> You must click 'Save ROI' before you can download ROIs</b> \
                </br> Scroll past the map to see the download progress \
                    <li> The data you downloaded will be in the 'data' folder</li> \
                    ",layout=Layout(margin='0px 0px 0px 5px'))

        self.instr_load_btns=HTML(
            value="<h2><b>Load ROIs/BBox</b></h2>\
                You can upload ROIs or Bbox geojson file.\
                    <li> Load BBox: Load bounding box from geojson file (ex. 'bbox.geojson')</li>\
                <li> Load ROIs: Load ROIs from geojson file (ex. 'rois.geojson')</li>\
                    ",layout=Layout(margin='0px 5px 0px 5px')) #top right bottom left

    def create_dashboard(self):
        """creates a dashboard containing all the buttons, instructions and widgets organized together.
        """
        save_vbox = VBox([self.instr_save_roi,self.save_button,self.save_bbox_button, self.instr_load_btns, self.load_rois_button, self.load_bbox_button])
        download_vbox = VBox([self.instr_download_roi,self.download_button])

        slider_v_box = VBox([self.small_fishnet_slider, self.large_fishnet_slider])
        slider_btn_box = HBox([slider_v_box, self.gen_button])
        roi_controls_box = VBox([self.instr_create_roi, slider_btn_box],layout=Layout(margin='0px 5px 5px 0px'))

        load_buttons = HBox([self.transects_button, self.shoreline_button])
        erase_buttons = HBox([self.remove_all_button, self.remove_transects_button, self.remove_bbox_button, self.remove_shoreline_button, self.remove_rois_button])


        row_1 = HBox([roi_controls_box,save_vbox,download_vbox])
        row_2 = HBox([load_buttons])
        row_3 = HBox([erase_buttons])
        row_4 = HBox([self.clear_debug_button, UI.debug_view])
        row_5 = HBox([self.coastseg_map.map])
        row_6 = HBox([UI.download_view])

        return display(row_1,row_2, row_3, row_4, row_5, row_6)

    def handle_small_slider_change(self, change):
        self.fishnet_sizes['small']=change['new']
    def handle_large_slider_change(self, change):
        self.fishnet_sizes['large']=change['new']


    @debug_view.capture(clear_output=True)
    def on_gen_button_clicked(self, btn):
        if self.coastseg_map.shapes_list == [] :
            with Tkinter_Window_Creator():
                messagebox.showinfo("Bounding Box Error", "Draw a bounding box on the coast first, then click Generate ROI.")
        else:
            UI.debug_view.clear_output(wait=True)
            self.coastseg_map.map.default_style = {'cursor': 'wait'}
            print("Generating ROIs please wait.")
            # Generate ROIs along the coastline within the bounding box
            self.coastseg_map.generate_ROIS_fishnet(self.fishnet_sizes['large'],self.fishnet_sizes['small'])
            # Add the Clickable ROIs to the map
            self.coastseg_map.add_geojson_layer_to_map()
            print("ROIs generated. Please Select at least one ROI and click Save ROI.")
            self.coastseg_map.map.default_style = {'cursor': 'default'}

    @debug_view.capture(clear_output=True)
    def on_transects_button_clicked(self,btn):
        if self.coastseg_map.shapes_list == [] :
            with Tkinter_Window_Creator():
                messagebox.showinfo("Bounding Box Error","Draw a bounding box on the coast first, then click Load Transects.")
        else:
            UI.debug_view.clear_output(wait=True)
            self.coastseg_map.map.default_style = {'cursor': 'wait'}
            print("Loading transects please wait.")
            # Add the transects to the map
            self.coastseg_map.load_transects_on_map()
            print("Transects Loaded.")
            self.coastseg_map.map.default_style = {'cursor': 'default'}


    @debug_view.capture(clear_output=True)
    def on_load_rois_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(),
                filetypes=[('geojson','*.geojson')],
                title = "Select a geojson file containing rois")
            # Save the filename as an attribute of the button
            if tk_root.filename:
                roi_file= tk_root.filename
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


    @debug_view.capture(clear_output=True)
    def on_shoreline_button_clicked(self,btn):
        self.coastseg_map.map.default_style = {'cursor': 'wait'}
        UI.debug_view.clear_output(wait=True)
        print("Loading shoreline please wait.")
        # Add the transects to the map
        try:
            self.coastseg_map.load_shoreline_on_map()
        except exceptions.BBox_Not_Found as bbox_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Bounding Box Error", "Draw a bounding box on the coast first, then click Load Transects.")
        except exceptions.Shoreline_Not_Found as shoreline_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Bounding Box Error", str(shoreline_error))
        else: 
            print("Shoreline loaded.")
        finally:
            self.coastseg_map.map.default_style = {'cursor': 'default'}


    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        if self.coastseg_map.selected_ROI:
            self.coastseg_map.map.default_style = {'cursor': 'wait'}
            UI.debug_view.append_stdout("Scroll down past map to see download progress.")
            try:
                self.download_button.disabled=True
                download_roi.download_imagery(self.coastseg_map.selected_ROI,
                                                self.coastseg_map.preprocess_settings,
                                                self.coastseg_map.dates,
                                                self.coastseg_map.sat_list,
                                                self.coastseg_map.collection)
            except google_auth_exceptions.RefreshError as exception:
                print(exception)
                with Tkinter_Window_Creator():
                    messagebox.showerror("Authentication Error", "Please authenticate with Google using the cell above: \n  'Authenticate and Initialize with Google Earth Engine (GEE)'")
            except Exception as exception2:
                print(exception2)
        else:
            UI.debug_view.append_stdout("No ROIs were selected. \nPlease select at least one ROI and click 'Save ROI' to save these ROI for download.")
            with Tkinter_Window_Creator():
                messagebox.showerror("ROI Selection Error", "No ROIs were selected. \nPlease select at least one ROI and click 'Save ROI' to save these ROI for download.")
        self.download_button.disabled=False
        self.coastseg_map.map.default_style = {'cursor': 'default'}

    @debug_view.capture(clear_output=True)
    def on_save_bbox_button_clicked(self, btn):
        if self.coastseg_map.shapes_list != [] :
            UI.debug_view.clear_output(wait=True)
            # Save selected bbox to a geojson file
            self.coastseg_map.save_bbox_to_file()
            UI.debug_view.clear_output(wait=True)
            print("BBox have been saved. Saved to bbox.geojson")
        else:
            with Tkinter_Window_Creator():
                messagebox.showerror("Bounding Box Error", "No bounding box found.\nDraw a bounding box on the coast first")

    @debug_view.capture(clear_output=True)
    def on_save_button_clicked(self,btn):
        if self.coastseg_map.selected_set:
            if len(self.coastseg_map.selected_set) == 0:
                with Tkinter_Window_Creator():
                    messagebox.showerror("ROI Selection Error", "Must select at least 1 ROI first before you can save ROIs.")
            else:
                UI.debug_view.clear_output(wait=True)
                self.coastseg_map.save_roi_fishnet("fishnet_rois.geojson")
                print("Saving ROIs")
                UI.debug_view.clear_output(wait=True)
                print("ROIs have been saved. Now click Download ROI to download the ROIs using CoastSat")
        else:
            with Tkinter_Window_Creator():
                messagebox.showerror("ROI Selection Error", "No ROIs were selected.")
            
    @debug_view.capture(clear_output=True)
    def on_load_bbox_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(),
                                                    filetypes=[('geojson','*.geojson')],
                                                    title = "Select a geojson file containing bbox")
            # Save the filename as an attribute of the button
            if tk_root.filename:
                bbox_file= tk_root.filename
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
                self.coastseg_map.map.add_layer(bbox_layer)
                print(f"Loaded the rois from the file :\n{bbox_file} ")
            else:
                messagebox.showerror("File Selection Error", "You must select a valid geojson file first!")

    @debug_view.capture(clear_output=True)
    def on_save_button_clicked(self, btn):
        if self.coastseg_map.selected_set:
            if len(self.coastseg_map.selected_set) == 0:
                with Tkinter_Window_Creator():
                    messagebox.showinfo("ROI Selection Error", "Must select at least 1 ROI first before you can save ROIs.")
            else:
                UI.debug_view.clear_output(wait=True)
                self.coastseg_map.save_roi_fishnet("fishnet_rois.geojson")
                print("Saving ROIs")
                UI.debug_view.clear_output(wait=True)
                print("ROIs have been saved. Now click Download ROI to download the ROIs using CoastSat")
        else:
            with Tkinter_Window_Creator():
                messagebox.showerror("ROI Selection Error", "No ROIs were selected.")

    def remove_all_from_map(self, btn):
        self.coastseg_map.remove_all()
    def remove_transects(self, btn):
        self.coastseg_map.remove_transects()
    def remove_bbox_from_map(self, btn):
        self.coastseg_map.remove_bbox()
    def remove_shoreline_from_map(self, btn):
        self.coastseg_map.remove_layer_by_name('shoreline')
    def remove_all_rois_from_map(self, btn):
        self.coastseg_map.remove_all_rois()
    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()
        UI.download_view.clear_output()