from typing import Callable
import ipywidgets
from ipyleaflet import GeoJSON
import logging


logger = logging.getLogger(__name__)

"""
This class is a widget that allows the user to load extracted shorelines on the map.
"""
# write docstring for this class
class Extracted_Shoreline_widget(ipywidgets.VBox):
    def __init__(self, map_interface=None):
        # map interface that has extracted shorelines
        self.map_interface = map_interface
        self.roi_html = ipywidgets.HTML(value="<b>ROI</b>:  ")
        self.status = ipywidgets.HTML(value="<b>Extracted Shoreline Status</b>:  ")
        self.satellite_html = ipywidgets.HTML(value="<b>Satellite</b>:  ")
        self.date_html = ipywidgets.HTML(value="<b>Date</b>:  ")
        title_html = ipywidgets.HTML(value="<h3>Load Extracted Shorelines</h3>")
        self.extracted_shoreline_slider = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=1,
            step=1,
            description="Load Shoreline:",
            disabled=True,
            continuous_update=False,  # only load in new value when slider is released
            orientation="horizontal",
        )
        self.load_extracted_shorelines_button = ipywidgets.Button(
            description="Load Shorelines"
        )

        # list of objects to watch
        self._observables = []
        self.extracted_shoreline_slider.observe(self.on_slider_change, names="value")
        # Roi information bar
        roi_info_row = ipywidgets.HBox([self.roi_html,self.satellite_html,self.date_html])
        super().__init__(
            [
                title_html,
                self.load_extracted_shorelines_button,
                self.status,
                self.extracted_shoreline_slider,
                ipywidgets.HTML(value="<b>Extracted Shoreline Information</b>:  "),
                roi_info_row,
            ]
        )

    def set_load_extracted_shorelines_button_on_click(self, on_click: Callable):
        logger.info(f"{on_click}")
        self.load_extracted_shorelines_button.on_click(lambda button: on_click())

    def set_max_slider(self, max: int):
        self.extracted_shoreline_slider.max = max

    def on_slider_change(self, change):
        # get the row number from the extracted_shoreline_slider
        row_number = change["new"]
        # get the extracted shoreline by the row number from the map_interface
        self.map_interface.load_extracted_shorelines_to_map(row_number=row_number)
        # update roi_html, satellite_html, date_html with the extracted shoreline information
        self.update(self._observables[-1])

    def set_roi_html(self, roi: str):
        self.roi_html.value = f"<b>ROI</b>: {roi} "

    def set_status_html(self, status: str):
        self.status.value = f"<b>Status</b>: {status} "

    def set_satellite_html(self, satellite: str):
        self.satellite_html.value = f"<b>Satellite</b>: {satellite} "

    def set_date_html(self, date: str):
        self.date_html.value = f"<b>Date</b>: {date} "

    def watch(self, observable):
        self._observables.append(observable)
        self._observables[-1].add_observer(self)
        self.update(self._observables[-1])

    def update(self, observable):
        logger.info(f"observable.name {observable.name} observable.get(): {observable.get()}")
        if observable.get() is not None:
            if isinstance(observable.get(), int):
                self.extracted_shoreline_slider.max = observable.get()
            if isinstance(observable.get(), str):
                if observable.name == "selected_roi_id":
                    self.set_roi_html(observable.get())
                if observable.name == "status":
                    self.set_status_html(observable.get())
            if isinstance(observable.get(), GeoJSON):
                self.extracted_shoreline_slider.disabled = False
                self.set_satellite_html(
                    satellite=observable.get().data["properties"]["satname"]
                )
                self.set_date_html(date=observable.get().data["properties"]["date"])
        else:
            self.extracted_shoreline_slider.max = 1
            self.extracted_shoreline_slider.disabled = True
            self.set_roi_html(roi="")
            self.set_satellite_html(satellite="")
            self.set_date_html(date="")
            self.set_status_html("")
