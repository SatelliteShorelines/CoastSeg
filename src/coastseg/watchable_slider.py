from typing import Callable
import logging

import ipywidgets
from ipyleaflet import GeoJSON
from ipywidgets import Layout


logger = logging.getLogger(__name__)

"""
This class is a widget that allows the user to load extracted shorelines on the map.
"""


# write docstring for this class
class Extracted_Shoreline_widget(ipywidgets.VBox):
    def __init__(self, map_interface=None):
        # map interface that has extracted shorelines
        self.roi_ids_with_shorelines = []
        self.map_interface = map_interface
        self.select_roi_dropdown = ipywidgets.Dropdown(
            options=[""],
            description="Select ROI:",
            style={"description_width": "initial"},
        )
        self.satellite_html = ipywidgets.HTML(value="<b>Satellite</b>:  ")
        self.date_html = ipywidgets.HTML(value="<b>Date</b>:  ")
        title_html = ipywidgets.HTML(
            value="<h3>Load Extracted Shorelines</h3>", layout=Layout(padding="0px")
        )
        self.extracted_shoreline_slider = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=1,
            step=1,
            description="Shoreline:",
            disabled=True,
            continuous_update=False,  # only load in new value when slider is released
            orientation="horizontal",
        )
        self.load_extracted_shorelines_button = ipywidgets.Button(
            description="Load Shorelines"
        )

        # list of objects to watch
        self._observables = []
        # list of functions to call when an observable changes
        self.extracted_shoreline_slider.observe(self.on_slider_change, names="value")
        self.select_roi_dropdown.observe(self.on_dropdown_change, names="value")
        # Roi information bar
        roi_info_row = ipywidgets.HBox([self.satellite_html, self.date_html])
        super().__init__(
            [
                title_html,
                self.select_roi_dropdown,
                self.extracted_shoreline_slider,
                ipywidgets.HTML(value="<b>Extracted Shoreline Information</b>:  "),
                roi_info_row,
            ]
        )

    def set_roi_ids_with_shorelines(self, roi_ids_with_shorelines: list):
        self.roi_ids_with_shorelines = roi_ids_with_shorelines
        self.select_roi_dropdown.options = self.roi_ids_with_shorelines

    def get_roi_ids_with_shorelines(self):
        return self.roi_ids_with_shorelines

    def set_load_extracted_shorelines_button_on_click(self, on_click: Callable):
        logger.info(f"{on_click}")
        self.load_extracted_shorelines_button.on_click(lambda button: on_click())

    def set_max_slider(self, max: int):
        self.extracted_shoreline_slider.max = max

    def on_slider_change(self, change):
        # get the row number from the extracted_shoreline_slider
        row_number = change["new"]
        # get the extracted shoreline by the row number from the map_interface
        roi_id = self.select_roi_dropdown.value
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
        if observable.name == "roi_ids_with_shorelines":
            if observable.get() is not None:
                self.set_roi_ids_with_shorelines(observable.get())
            elif observable.get() is None:
                self.set_roi_ids_with_shorelines([""])
        if observable.name == "number_extracted_shorelines":
            if observable.get() is not None:
                self.extracted_shoreline_slider.max = observable.get()
            elif observable.get() is None:
                self.extracted_shoreline_slider.max = 1
        if observable.name == "extracted_shoreline_layer":
            if observable.get() is not None:
                self.extracted_shoreline_slider.disabled = False
                self.set_satellite_html(
                    satellite=observable.get().data["properties"]["satname"]
                )
                self.set_date_html(date=observable.get().data["properties"]["date"])
            elif observable.get() is None:
                self.extracted_shoreline_slider.disabled = True
                self.set_satellite_html(satellite="")
                self.set_date_html(date="")
