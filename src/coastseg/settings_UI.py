# standard python imports
# external python imports
import ipywidgets
from ipywidgets import VBox
from ipywidgets import HTML
from ipywidgets import BoundedFloatText
from ipywidgets import Text
from ipywidgets import Select
from ipywidgets import BoundedIntText


class Settings_UI:
    def __init__(self) -> None:
        self.settings = {}
        # button styles
        self.remove_style = dict(button_color="red")
        self.load_style = dict(button_color="#69add1", description_width="initial")
        self.action_style = dict(button_color="#ae3cf0")
        self.save_style = dict(button_color="#50bf8f")
        self.clear_stlye = dict(button_color="#a3adac")

    def set_settings(self, **kwargs):
        self.settings.update({key: value for key, value in kwargs.items()})

    def get_settings(self) -> dict:
        settings = {
            "max_dist_ref": self.shoreline_buffer_slider.value,
            "along_dist": self.alongshore_distance_slider.value,
            "dist_clouds": self.cloud_slider.value,
            "min_beach_area": self.beach_area_slider.value,
            "min_length_sl": self.min_length_sl_slider.value,
            "cloud_thresh": self.cloud_threshold_slider.value,
            "min_points": self.min_points_text.value,
            "max_std": self.max_std_text.value,
            "max_range": self.max_range_text.value,
            "min_chainage": self.min_chainage_text.value,
            "multiple_inter": self.outliers_mode.value,
            "prc_multiple": self.prc_multiple_text.value,
            "percent_no_data": self.no_data_slider.value,
        }
        self.set_settings(**settings)

        return self.settings

    def get_no_data_slider(self):
        # returns slider to control no data slider
        instructions = HTML(value="<b>Percentage of Bad Pixels Allowed</b>")

        self.no_data_slider = ipywidgets.FloatSlider(
            value=50.0,
            min=0.0,
            max=100.0,
            step=1.0,
            description="Percentage Bad Pixels:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        return VBox([instructions, self.no_data_slider])

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

    def render(self):
        # return a vbox of all the settings
        settings = {
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
            "percent_no_data": self.get_no_data_slider(),
        }
        # create settings vbox
        settings_vbox = VBox([widget for widget_name, widget in settings.items()])
        return settings_vbox
