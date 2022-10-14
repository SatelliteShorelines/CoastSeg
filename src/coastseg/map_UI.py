# standard python imports
import os
import logging
#internal python imports
from src.coastseg.tkinter_window_creator import Tkinter_Window_Creator
from src.coastseg import exceptions
# external python imports
import ipywidgets
from IPython.display import display, clear_output
from google.auth import exceptions as google_auth_exceptions
from tkinter import filedialog
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

logger = logging.getLogger(__name__)
logger.info("I am a log from %s",__name__)

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
        self.load_gdf_button = Button(description="Load gdf from file",style=self.load_style)
        self.load_gdf_button.on_click(self.on_load_gdf_clicked)
        self.load_bbox_button = Button(description="Load bbox from file",style=self.load_style)
        self.load_bbox_button.on_click(self.on_load_bbox_clicked)
        # buttons to load configuration files
        self.load_configs_button = Button(description="Load configs",style=self.load_style)
        self.load_configs_button.on_click(self.on_load_configs_clicked)
        self.save_config_button = Button(description="Save config files",style=self.save_style)
        self.save_config_button.on_click(self.on_save_config_clicked)
        # load buttons
        self.transects_button = Button(description="Load Transects",style=self.load_style)
        self.transects_button.on_click(self.on_transects_button_clicked)
        self.shoreline_button = Button(description="Load Shoreline",style=self.load_style)
        self.shoreline_button.on_click(self.on_shoreline_button_clicked)
        self.load_rois_button = Button(description="Load rois from file",style=self.load_style)
        self.load_rois_button.on_click(self.on_load_rois_clicked)
        # Save buttons
        self.save_shoreline_button = Button(description="Save shorelines",style=self.save_style)
        self.save_shoreline_button.on_click(self.save_shoreline_button_clicked)
        self.save_transects_button = Button(description="Save transects",style=self.save_style)
        self.save_transects_button.on_click(self.save_transects_button_clicked)
        self.save_roi_button = Button(description="Save ROI",style=self.save_style)
        self.save_roi_button.on_click(self.save_roi_button_clicked)
        self.save_bbox_button = Button(description="Save Bbox",style=self.save_style)
        self.save_bbox_button.on_click(self.on_save_bbox_button_clicked)
        # Generate buttons
        self.gen_button = Button(description="Generate ROI",style=self.action_style)
        self.gen_button.on_click(self.on_gen_button_clicked)
        self.download_button = Button(description="Download ROIs",style=self.action_style)
        self.download_button.on_click(self.download_button_clicked)
        self.extract_shorelines_button = Button(description="Extract Shorelines",style=self.action_style)
        self.extract_shorelines_button.on_click(self.extract_shorelines_button_clicked)
        self.compute_transect_button = Button(description="Compute Transects",style=self.action_style)
        self.compute_transect_button.on_click(self.compute_transect_button_clicked)
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
        
        self.instr_config_btns=HTML(
            value="<h2><b>Load Config Files</b></h2>\
                Use the load configs button to select a config.geojson file.\
                Make sure the config.json is in the same directory\
                You can upload a config json and config geojson file.\
                    <li> Load Config Json: Load settings from json file (ex. 'config_id_241.json')</li>\
                <li> Load Config geojson: Load rois, shorelines, and transects from geojson file (ex. 'config_gdf_id_241.geojson')</li>\
                    ",layout=Layout(margin='0px 5px 0px 5px')) #top right bottom left

    def create_dashboard(self):
        """creates a dashboard containing all the buttons, instructions and widgets organized together.
        """
        save_vbox = VBox([self.instr_save_roi,self.save_roi_button,self.save_bbox_button,self.save_shoreline_button, self.save_transects_button, self.instr_load_btns, self.load_rois_button, self.load_bbox_button,self.load_gdf_button])
        config_vbox = VBox([self.instr_config_btns, self.load_configs_button, self.save_config_button])
        download_vbox = VBox([self.instr_download_roi,self.download_button,self.extract_shorelines_button, self.compute_transect_button,config_vbox])

        slider_v_box = VBox([self.small_fishnet_slider, self.large_fishnet_slider])
        slider_btn_box = VBox([slider_v_box, self.gen_button])
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
        UI.debug_view.clear_output(wait=True)
        try:
            print("Generating ROIs please wait.")
            self.coastseg_map.map.default_style = {'cursor': 'wait'}
            # Generate ROIs along the coastline within the bounding box
            self.coastseg_map.load_rois_on_map(self.fishnet_sizes['large'],self.fishnet_sizes['small'])
            # self.coastseg_map.generate_ROIS_fishnet(self.fishnet_sizes['large'],self.fishnet_sizes['small'])
        except exceptions.Object_Not_Found as not_on_map_error:
            with Tkinter_Window_Creator():
                messagebox.showwarning("Bounding Box Error", f'{not_on_map_error}')
        except Exception as exception:
            with Tkinter_Window_Creator():
                messagebox.showwarning("Error", f'{exception}')
        else:
            print("ROIs generated. Please Select at least one ROI and click Save ROI.")
        finally:
            self.coastseg_map.map.default_style = {'cursor': 'default'}
        

    @debug_view.capture(clear_output=True)
    def on_transects_button_clicked(self,btn):
        UI.debug_view.clear_output(wait=True)
        try:
            print("Loading transects please wait.")
            self.coastseg_map.map.default_style = {'cursor': 'wait'}
            self.coastseg_map.load_transects_on_map()
        except exceptions.Object_Not_Found as not_on_map_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Bounding Box Error", str(not_on_map_error))    
        else:
            print("Transects Loaded.")
        finally:
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
                try:
                    self.coastseg_map.load_rois_on_map(file=tk_root.filename)
                except exceptions.Object_Not_Found as not_on_map_error:
                    with Tkinter_Window_Creator():
                        messagebox.showinfo("Error", str(not_on_map_error))
                except Exception as error:
                    with Tkinter_Window_Creator():
                        messagebox.showinfo("Error", str(error))        
            else:
                messagebox.showerror("ROI Selection Error", "You must select a valid geojson file first!")


    @debug_view.capture(clear_output=True)
    def on_shoreline_button_clicked(self,btn):
        self.coastseg_map.map.default_style = {'cursor': 'wait'}
        UI.debug_view.clear_output(wait=True)
        print("Loading shoreline please wait...")
        try:
            self.coastseg_map.load_shoreline_on_map()
        except exceptions.Object_Not_Found as not_on_map_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Bounding Box Error", str(not_on_map_error))
        except Exception as error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(error))    
        else: 
            print("Shoreline loaded.")
        finally:
            self.coastseg_map.map.default_style = {'cursor': 'default'}

    @debug_view.capture(clear_output=True)
    def extract_shorelines_button_clicked(self,btn):
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {'cursor': 'wait'}
        try:
            self.extract_shorelines_button.disabled=True
            self.coastseg_map.extract_all_shorelines()
        except Exception as exception:
            with Tkinter_Window_Creator():
                messagebox.showerror("Error", f"{exception}" )
        finally:
            self.extract_shorelines_button.disabled=False
            self.coastseg_map.map.default_style = {'cursor': 'default'}
            
    @debug_view.capture(clear_output=True)
    def compute_transect_button_clicked(self,btn):
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {'cursor': 'wait'}
        try:
            self.compute_transect_button.disabled=True
            self.coastseg_map.compute_transects()
        except Exception as exception:
            with Tkinter_Window_Creator():
                messagebox.showerror("Error", f"{exception}" )
        finally:
            self.compute_transect_button.disabled=False
            self.coastseg_map.map.default_style = {'cursor': 'default'}

    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {'cursor': 'wait'}
        UI.debug_view.append_stdout("Scroll down past map to see download progress.")
        try:
            self.download_button.disabled=True
            self.coastseg_map.download_imagery()
        except google_auth_exceptions.RefreshError as exception:
            print(exception)
            with Tkinter_Window_Creator():
                messagebox.showerror("Authentication Error", "Please authenticate with Google using the cell above: \n  'Authenticate and Initialize with Google Earth Engine (GEE)'")
        except Exception as exception:
            with Tkinter_Window_Creator():
                messagebox.showerror("Error", f"{exception}" )
        
        self.download_button.disabled=False
        self.coastseg_map.map.default_style = {'cursor': 'default'}
    
    @debug_view.capture(clear_output=True)
    def save_transects_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(self.coastseg_map.transects)
        except exceptions.Object_Not_Found as not_on_map_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(not_on_map_error))
        except Exception as error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(error)) 

    @debug_view.capture(clear_output=True)
    def save_shoreline_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(self.coastseg_map.shoreline)
        except exceptions.Object_Not_Found as not_on_map_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(not_on_map_error))
        except Exception as error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(error)) 


    @debug_view.capture(clear_output=True)
    def on_save_bbox_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(self.coastseg_map.bbox)
        except exceptions.Object_Not_Found as not_on_map_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(not_on_map_error))
        except Exception as error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(error)) 


    @debug_view.capture(clear_output=True)
    def on_load_gdf_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(),
                                                    filetypes=[('geojson','*.geojson')],
                                                    title = "Select a geojson file")
            # Save the filename as an attribute of the button
            if tk_root.filename:
                self.coastseg_map.load_gdf_on_map(tk_root.filename)
            else:
                messagebox.showerror("File Selection Error", "You must select a valid geojson file first!")


    @debug_view.capture(clear_output=True)
    def on_load_configs_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(),
                                                    filetypes=[('geojson','*.geojson')],
                                                    title = "Select a geojson file")
            # Save the filename as an attribute of the button
            try:
                if tk_root.filename:
                    self.coastseg_map.load_configs(tk_root.filename)
                else:
                    messagebox.showerror("File Selection Error", "You must select a valid geojson file first!")
            except Exception as error:
                messagebox.showinfo("Error", str(error))
                
    @debug_view.capture(clear_output=True)
    def on_save_config_clicked(self, button):
        try:
            self.coastseg_map.save_config(self.coastseg_map.data_downloaded)
            # self.coastseg_map.save_config(self.coastseg_map.data_downloaded, os.getcwd())
        except Exception as error:
            with Tkinter_Window_Creator():
                logger.error(error)
                messagebox.showinfo("Error", str(error))


    @debug_view.capture(clear_output=True)
    def on_load_bbox_clicked(self, button):
        # Prompt the user to select a directory of images
        with Tkinter_Window_Creator() as tk_root:
            tk_root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(),
                                                    filetypes=[('geojson','*.geojson')],
                                                    title = "Select a geojson file containing bbox")
            # Save the filename as an attribute of the button
            if tk_root.filename:
                self.coastseg_map.load_bbox_on_map(tk_root.filename)
            else:
                messagebox.showerror("File Selection Error", "You must select a valid geojson file first!")


    @debug_view.capture(clear_output=True)
    def save_roi_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.coastseg_map.save_feature_to_file(self.coastseg_map.rois)
            # UI.debug_view.clear_output(wait=True)
            print("ROIs have been saved. Now click Download ROI to download the ROIs using CoastSat")
        except exceptions.Object_Not_Found as not_on_map_error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("Error", str(not_on_map_error))
        except Exception as error:
            with Tkinter_Window_Creator():
                messagebox.showinfo("ROI Selection Error", str(error))
                        
    def remove_all_from_map(self, btn):
        self.coastseg_map.remove_all()
    def remove_transects(self, btn):
        self.coastseg_map.remove_transects()
    def remove_bbox_from_map(self, btn):
        self.coastseg_map.remove_bbox()
    def remove_shoreline_from_map(self, btn):
        self.coastseg_map.remove_shoreline()
    def remove_all_rois_from_map(self, btn):
        self.coastseg_map.remove_all_rois()
    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()
        UI.download_view.clear_output()