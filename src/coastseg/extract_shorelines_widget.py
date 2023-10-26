import logging
from typing import Callable, List
import ipywidgets
from ipywidgets import Layout


logger = logging.getLogger(__name__)


class Extracted_Shoreline_widget(ipywidgets.VBox):
    """
    A widget that allows user to control the extracted shorelines loaded on the map and delete them.

    Parameters
    ----------
    extracted_shoreline_traitlet : traitlets.Traitlet
        A traitlet that holds the list of extracted shorelines available to load.

    Attributes
    ----------
    compact_layout : ipywidgets.Layout
        A layout for the widget.
    button_layout : ipywidgets.Layout
        A layout for the buttons.

    Methods
    -------
    add_load_callback(callback: Callable[[List[str]], None])
        Add a callback function to be called when a shoreline is selected.
    add_remove_callback(callback: Callable[[List[str]], None])
        Add a callback function to be called when shorelines need to be removed.
    add_remove_all_callback(callback: Callable[[List[str]], None])
        Add a callback function to be called when a the remove all button is clicked.
    on_load_selected(change:dict)
        Callback function for when a load shoreline item is selected.
    on_trash_selected(change:dict)
        Callback function for when a trash item is selected.
    trash_button_clicked(btn)
        Callback function for when the trash button is clicked.
    undo_button_clicked(btn)
        Callback function for when the undo button is clicked.
    delete_all_button_clicked(btn)
        Callback function for when the delete all button is clicked.
    """

    # Define the button layout
    button_layout = ipywidgets.Layout(
        width="auto",
        height="25%",
        margin="5px",
        padding="0px",
    )

    # Define the compact layout for the VBoxes
    compact_layout = ipywidgets.Layout(
        width="70%", overflow="auto", height="auto", padding="0px", margin="0px"
    )

    def __init__(self, extracted_shoreline_traitlet):
        """
        Initializes the ExtractShorelinesWidget.

        Parameters
        ----------
        extracted_shoreline_traitlet : traitlets.List
            A traitlet that holds the extracted shorelines.
        """
        # 1. Initialization of Basic Attributes
        self.extracted_shoreline_traitlet = extracted_shoreline_traitlet
        self.remove_callback = None
        self.load_callback = None

        # 2. Widget Creation

        # List widgets
        self.load_list_widget = ipywidgets.SelectMultiple(
            description="",
            options=[],
            layout=ipywidgets.Layout(padding="0px", margin="0px"),
        )
        self.trash_list_widget = ipywidgets.SelectMultiple(
            description="",
            options=[],
            layout=ipywidgets.Layout(padding="0px", margin="0px"),
        )
        self.ROI_list_widget = ipywidgets.Dropdown(
            description="Available ROIs",
            options=[],
            layout=ipywidgets.Layout(padding="0px", margin="0px"),
        )

        # Buttons
        self.load_trash_button = ipywidgets.Button(
            description="",
            icon="trash",
            button_style="danger",
            layout=ipywidgets.Layout(width="8%", height="30%", margin="10px"),
            tooltip="Trash",
        )
        self.empty_trash_button = ipywidgets.Button(
            description="Empty Trash",
            icon="trash",
            button_style="danger",
            layout=Extracted_Shoreline_widget.button_layout,
            tooltip="Empty Trash",
        )
        self.undo_button = ipywidgets.Button(
            description="",
            icon="undo",
            layout=Extracted_Shoreline_widget.button_layout,
            tooltip="Undo",
        )

        # Instructions and Title
        load_instruction = ipywidgets.HTML(
            value="<div style='white-space: normal; word-wrap: break-word;'> Hold <code>Ctrl</code> to select multiple shorelines. Click the trash icon to move to trash.</div>"
        )
        trash_instruction = ipywidgets.HTML(
            value="<div style='white-space: normal; word-wrap: break-word;'> Hold <code>Ctrl</code> to select multiple shorelines. Click the empty trash to delete or undo to restore.</div>"
        )
        title_html = ipywidgets.HTML(
            value="<h3>Load Extracted Shorelines</h3>", layout=Layout(padding="0px")
        )

        # 3. Configuration and Event Handling
        # add on click handlers for the buttons
        self.load_trash_button.on_click(self.trash_button_clicked)
        self.undo_button.on_click(self.undo_button_clicked)
        self.empty_trash_button.on_click(self.delete_all_button_clicked)

        # callback function for when a load shoreline item is selected
        self.load_list_widget.observe(self.on_load_selected, names="value")
        # callback function for when a trash item is selected
        self.trash_list_widget.observe(self.on_trash_selected, names="value")
        # artifically select the first value when any shoreline is loaded
        if self.load_list_widget.options:
            self.load_list_widget.value = (self.load_list_widget.options[0],)

        # 4. Layout Assembly
        load_list_vbox = ipywidgets.HBox(
            [self.load_list_widget, self.load_trash_button]
        )
        trash_list_vbox = ipywidgets.HBox(
            [
                self.trash_list_widget,
                ipywidgets.VBox(
                    [self.undo_button, self.empty_trash_button],
                    layout=ipywidgets.Layout(width="30%"),
                ),
            ],
            layout=ipywidgets.Layout(
                width="90%", overflow="visible", background_color="#f2f2f2"
            ),
        )
        load_instruction_box = ipywidgets.VBox(
            [load_instruction], layout=Extracted_Shoreline_widget.compact_layout
        )
        trash_instruction_box = ipywidgets.VBox(
            [trash_instruction], layout=Extracted_Shoreline_widget.compact_layout
        )

        total_VBOX = ipywidgets.VBox(
            [
                title_html,
                load_instruction_box,
                load_list_vbox,
                trash_instruction_box,
                trash_list_vbox,
            ],
            layout=ipywidgets.Layout(
                width="100%", overflow="visible", height="auto", padding="0px"
            ),
        )

        super().__init__([total_VBOX])

    # def add_ROI_callback(self, callback: Callable[[List[str]], None]):
    #     """
    #     Add a callback function to be called when a ROI ID is selected.

    #     Parameters
    #     ----------
    #     callback : function
    #         The function to be called when a ROI ID is selected.
    #     """
    #     self.roi_selected_callback = callback

    def add_load_callback(self, callback: Callable[[List[str]], None]):
        """
        Add a callback function to be called when a shoreline is selected.

        Parameters
        ----------
        callback : function
            The function to be called when a shoreline is selected.
        """
        self.load_callback = callback

    def add_remove_all_callback(self, callback: Callable[[List[str]], None]):
        """
        Add a callback function to be called when a the remove all button is clicked.

        Parameters
        ----------
        callback : function
            The function to be called when the remove all button is clicked.
        """
        self.remove_all_callback = callback

    def add_remove_callback(self, callback: Callable[[List[str]], None]):
        """
        Add a callback function to be called when shorelines need to be removed

        Parameters
        ----------
        callback : function
            The function to be called  when shorelines need to be removed.
        """
        self.remove_callback = callback

    # def on_roi_selected(self, change: dict):
    #     """Callback function for when an ROI ID is selected"""
    #     # when the content sof the load list changes update the layer
    #     # self.roi_selected_callback(change["new"])

    def on_load_selected(self, change: dict):
        """Callback function for when a load shoreline item is selected"""
        # when the content sof the load list changes update the layer
        style = {
            "color": "#001aff",  # Outline color
            "opacity": 1,  # opacity 1 means no transparency
            "weight": 3,  # Width
            "fillColor": "#001aff",  # Fill color
            "fillOpacity": 0.8,  # Fill opacity.
            "radius": 1,
        }
        layer_name = "extracted shoreline"
        selected_items = self.load_list_widget.value
        if selected_items and self.load_callback:
            self.load_callback(selected_items, layer_name, style)
        else:
            # if no items are selected remove the shorelines from the map
            if self.remove_callback:
                self.remove_callback(layer_name)

    def on_trash_selected(self, change: dict):
        """Callback function for when a trash item is selected"""
        # this creates new a layer on the map that's styled red
        style = {
            "color": "#e82009",  # Outline color
            "opacity": 1,  # opacity 1 means no transparency
            "weight": 3,  # Width
            "fillColor": "#e82009",  # Fill color
            "fillOpacity": 0.8,  # Fill opacity.
            "radius": 1,
        }
        selected_items = self.trash_list_widget.value
        layer_name = "delete"
        if selected_items and self.load_callback:
            self.load_callback(selected_items, layer_name, style)
        else:
            # if no items are selected remove the shorelines from the map
            if self.remove_callback:
                self.remove_callback(layer_name)

    def trash_button_clicked(self, btn):
        """Callback function for when the trash button is clicked"""
        # get selected shorelines out of all the available shorelines
        selected_items = self.load_list_widget.value
        print(selected_items)
        # add
        self.extracted_shoreline_traitlet.trash_list = (
            self.extracted_shoreline_traitlet.trash_list + list(selected_items)
        )
        # remove the items from the load_list
        self.extracted_shoreline_traitlet.load_list = list(
            set(self.extracted_shoreline_traitlet.load_list) - set(selected_items)
        )

    def undo_button_clicked(self, btn):
        """Callback function for when the undo button is clicked"""
        selected_items = self.trash_list_widget.value
        self.extracted_shoreline_traitlet.trash_list = list(
            set(self.extracted_shoreline_traitlet.trash_list) - set(selected_items)
        )
        # add the items back to the load_list
        self.extracted_shoreline_traitlet.load_list = list(
            self.extracted_shoreline_traitlet.load_list + list(selected_items)
        )

    def delete_all_button_clicked(self, btn):
        """Callback function for when the delete all button is clicked"""
        selected_items = self.trash_list_widget.value
        self.extracted_shoreline_traitlet.trash_list = list(
            set(self.trash_list_widget.options) - set(selected_items)
        )
        self.extracted_shoreline_traitlet.load_list = list(
            set(self.load_list_widget.options) - set(selected_items)
        )
        if self.remove_all_callback:
            layer_name = "delete"
            self.remove_all_callback(layer_name, selected_items)
