from typing import Callable
import logging

import ipywidgets
from ipywidgets import Layout


logger = logging.getLogger(__name__)

"""
This class is a widget that allows the user to load extracted shorelines on the map.
"""


# write docstring for this class
class Extracted_Shoreline_widget(ipywidgets.VBox):
    def __init__(self, map_interface=None):
        # map interface that has extracted shorelines
        self.map_interface = map_interface

        self.satellite_html = ipywidgets.HTML(value="<b>Satellite</b>:  ")
        self.date_html = ipywidgets.HTML(value="<b>Date</b>:  ")
        title_html = ipywidgets.HTML(
            value="<h3>Load Extracted Shorelines</h3>", layout=Layout(padding="0px")
        )

        self.create_dropdown()
        self.create_slider()

        self.load_extracted_shorelines_button = ipywidgets.Button(
            description="Load Shorelines"
        )

        # list of objects to watch
        self._observables = []
        # Roi information bar
        roi_info_row = ipywidgets.HBox([self.satellite_html, self.date_html])
        super().__init__(
            [
                title_html,
                self.dropdown,
                self.slider,
                ipywidgets.HTML(value="<b>Extracted Shoreline Information</b>:  "),
                roi_info_row,
            ]
        )

    def create_slider(self):
        self.slider = ipywidgets.IntSlider(
            value=self.map_interface.extract_shorelines_container.max_shorelines,
            min=0,
            max=1,
            step=1,
            description="Shoreline:",
            disabled=True,
            continuous_update=False,  # only load in new value when slider is released
            orientation="horizontal",
        )

        # Function to update widget options when the traitlet changes
        def update_extracted_shoreline_slider(change):
            self.slider.max = change["new"]

        # When the traitlet,id_container, trait 'ids' changes the update_extracted_shoreline_slider will be updated
        self.map_interface.extract_shorelines_container.observe(
            update_extracted_shoreline_slider, names="max_shorelines"
        )
        self.slider.observe(self.on_slider_change, names="value")

    def create_dropdown(self):
        self.dropdown = ipywidgets.Dropdown(
            options=self.map_interface.id_container.ids,
            description="Select ROI:",
            style={"description_width": "initial"},
        )

        # Function to update widget options when the traitlet changes
        def update_select_roi_dropdown(change):
            self.dropdown.options = change["new"]

        # When the traitlet,id_container, trait 'ids' changes the update_select_roi_dropdown will be updated
        self.map_interface.id_container.observe(update_select_roi_dropdown, names="ids")
        self.dropdown.observe(self.on_dropdown_change, names="value")

    def set_load_extracted_shorelines_button_on_click(self, on_click: Callable):
        logger.info(f"{on_click}")
        self.load_extracted_shorelines_button.on_click(lambda button: on_click())

    def on_slider_change(self, change):
        # get the row number from the extracted_shoreline_slider
        row_number = change["new"]
        # get the extracted shoreline by the row number from the map_interface
        roi_id = self.dropdown.value
        self.map_interface.load_extracted_shoreline_by_id(roi_id, row_number=row_number)
        # update roi_html, satellite_html, date_html with the extracted shoreline information
        self.update(self._observables[-1])

    def on_dropdown_change(self, change):
        roi_id = change["new"]
        # get the extracted shoreline by the row number from the map_interface
        self.map_interface.load_extracted_shoreline_by_id(roi_id, row_number=0)
        # update roi_html, satellite_html, date_html with the extracted shoreline information
        self.update(self._observables[-1])

    def set_satellite_html(self, satellite: str):
        self.satellite_html.value = f"<b>Satellite</b>: {satellite} "

    def set_date_html(self, date: str):
        self.date_html.value = f"<b>Date</b>: {date} "

    def watch(self, observable):
        self._observables.append(observable)
        self._observables[-1].add_observer(self)
        self.update(self._observables[-1])

    def update(self, observable):
        logger.info(
            f"observable.name {observable.name} observable.get(): {observable.get()}"
        )
        if observable.name == "extracted_shoreline_layer":
            if observable.get() is not None:
                self.slider.disabled = False
                self.set_satellite_html(
                    satellite=observable.get().data["properties"]["satname"]
                )
                self.set_date_html(date=observable.get().data["properties"]["date"])
            elif observable.get() is None:
                self.slider.disabled = True
                self.set_satellite_html(satellite="")
                self.set_date_html(date="")
