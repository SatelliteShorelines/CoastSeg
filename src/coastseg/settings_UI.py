# standard python imports
# external python imports
import ipywidgets
import datetime
from typing import List, Union, Optional, Tuple


class ButtonColors:
    REMOVE = "red"
    LOAD = "#69add1"
    ACTION = "#ae3cf0"
    SAVE = "#50bf8f"
    CLEAR = "#a3adac"


def str_to_bool(var: str) -> bool:
    return var == "True"


def convert_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date: {date_str}. Expected format: 'YYYY-MM-DD'.{e}")


class Settings_UI:
    def __init__(
        self,
        basic_settings: Optional[List[str]] = None,
        advanced_settings: Optional[List[str]] = None,
    ) -> None:
        # if no basic settings are provided, use the default settings
        if basic_settings is None:
            basic_settings = [
                "max_dist_ref",
                "min_length_sl",
                "min_beach_area",
                "dist_clouds",
                "apply_cloud_mask",
                "cloud_thresh",
                "percent_no_data",
            ]
        if advanced_settings is None:
            advanced_settings = [
                "min_points",
                "max_std",
                "along_dist",
                "max_range",
                "min_chainage",
                "multiple_inter",
                "prc_multiple",
            ]

        self.settings = {}
        self.basic_settings = basic_settings
        self.advanced_settings = advanced_settings
        self.settings_widgets = {}

        # button styles
        self.remove_style = dict(button_color=ButtonColors.REMOVE)
        self.load_style = dict(
            button_color=ButtonColors.LOAD, description_width="initial"
        )
        self.action_style = dict(button_color=ButtonColors.ACTION)
        self.save_style = dict(button_color=ButtonColors.SAVE)
        self.clear_stlye = dict(button_color=ButtonColors.CLEAR)

        # Create the basic settings tab
        self.basic_settings_tab = self.create_settings_tab(self.basic_settings)

        # Create the advanced settings tab
        self.advanced_settings_tab = self.create_settings_tab(self.advanced_settings)

    def create_settings_tab(self, settings: List[str]) -> ipywidgets.VBox:
        # Create the settings tab
        tab_contents = []
        for setting_name in settings:
            # Create the widget for the setting
            widget, instructions = self.create_setting_widget(setting_name)

            # Add the widget and instructions to the tab contents
            tab_contents.append(ipywidgets.VBox([instructions, widget]))

            # Add the widget to the settings_widgets dictionary
            self.settings_widgets[setting_name] = widget

        # Create the settings tab
        tab = ipywidgets.VBox(children=tab_contents)

        return tab

    def add_custom_widget(
        self,
        widget: Union[
            ipywidgets.ToggleButton, ipywidgets.FloatSlider, ipywidgets.IntText
        ],
        setting_name: str,
        title: str,
        instructions: str,
        advanced: bool = False,
        index: Optional[int] = None,
    ):
        """
        Adds a custom widget to the basic or advanced settings tab at the specified index.

        Args:
            widget: The widget to add.
            setting_name: The name of the setting.
            title: The title of the setting.
            instructions: Optional instructions for the widget.
            advanced: Whether to add the widget to the advanced settings tab. If False, adds to the basic settings tab.
            index: The index at which to insert the widget. If None, the widget is added to the end of the settings list.
        """
        # Check for missing title, setting_name, or instructions
        if not title:
            raise ValueError("Title cannot be empty.")
        if not setting_name:
            raise ValueError("Setting name cannot be empty.")
        if not instructions:
            instructions = ""
        # Add the widget to the settings tab
        if advanced:
            if index is None:
                index = len(self.advanced_settings)
            self.advanced_settings.insert(index, setting_name)
            self.settings_widgets[setting_name] = widget
            self.advanced_settings_tab.children = (
                self.advanced_settings_tab.children[:index]
                + (ipywidgets.HTML(value=f"<b>{title}</b><br>{instructions}."),)
                + (self.settings_widgets[setting_name],)
                + self.advanced_settings_tab.children[index:]
            )
        else:
            if index is None:
                index = len(self.basic_settings)
            self.basic_settings.insert(index, setting_name)
            self.settings_widgets[setting_name] = widget
            self.basic_settings_tab.children = (
                self.basic_settings_tab.children[:index]
                + (ipywidgets.HTML(value=f"<b>{title}</b><br>{instructions}."),)
                + (self.settings_widgets[setting_name],)
                + self.basic_settings_tab.children[index:]
            )

    def create_setting_widget(
        self, setting_name: str
    ) -> Tuple[
        Union[ipywidgets.ToggleButton, ipywidgets.FloatSlider, ipywidgets.IntText],
        ipywidgets.HTML,
    ]:
        # Create the widget for the setting
        if setting_name == "apply_cloud_mask":
            widget = ipywidgets.ToggleButtons(
                options=["True", "False"],
                description="Apply Cloud Mask",
                tooltips=[
                    "Cloud Masking On",
                    "Cloud Masking Off",
                ],
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Apply Cloud Mask</b><br>Enable/disable cloud masking."
            )
        elif setting_name == "max_dist_ref":
            widget = ipywidgets.IntSlider(
                description="Max Distance Reference",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Max Distance Reference</b><br>Maximum distance from the shoreline to search for reference points."
            )
        elif setting_name == "along_dist":
            widget = ipywidgets.IntSlider(
                description="Alongshore Distance",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Alongshore Distance</b><br>Distance along the shoreline to search for reference points."
            )
        elif setting_name == "dist_clouds":
            widget = ipywidgets.IntSlider(
                description="Distance to Clouds",
                min=0,
                step=1,
                max=1000,
                value=300,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Distance to Clouds</b><br>Maximum distance from the shoreline to search for clouds."
            )
        elif setting_name == "min_beach_area":
            widget = ipywidgets.IntSlider(
                description="Minimum Beach Area",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Beach Area</b><br>Minimum area of beach required to be considered a valid reference point."
            )
        elif setting_name == "min_length_sl":
            widget = ipywidgets.IntSlider(
                description="Minimum Shoreline Length",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Shoreline Length</b><br>Minimum length of shoreline required to be considered a valid reference point."
            )
        elif setting_name == "cloud_thresh":
            widget = ipywidgets.IntSlider(
                description="Cloud Threshold",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Cloud Threshold</b><br>Threshold for cloud detection."
            )
        elif setting_name == "min_points":
            widget = ipywidgets.IntText(
                description="Minimum Points",
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Points</b><br>Minimum number of reference points required to calculate shoreline."
            )
        elif setting_name == "max_std":
            widget = ipywidgets.IntSlider(
                description="Maximum Standard Deviation",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Maximum Standard Deviation</b><br>Maximum standard deviation allowed for reference points."
            )
        elif setting_name == "max_range":
            widget = ipywidgets.IntSlider(
                description="Maximum Range",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Maximum Range</b><br>Maximum range allowed for reference points."
            )
        elif setting_name == "min_chainage":
            widget = ipywidgets.IntSlider(
                description="Minimum Chainage",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Chainage</b><br>Minimum chainage required to be considered a valid reference point."
            )
        elif setting_name == "multiple_inter":
            widget = ipywidgets.ToggleButton(
                description="Multiple Intersections",
                value=True,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Multiple Intersections</b><br>Enable/disable multiple intersection detection."
            )
        elif setting_name == "prc_multiple":
            widget = ipywidgets.FloatSlider(
                description="Percentage Multiple Intersections",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Percentage Multiple Intersections</b><br>Percentage of multiple intersections allowed."
            )
        elif setting_name == "percent_no_data":
            widget = ipywidgets.FloatSlider(
                description="Percentage of Bad Pixels Allowed",
                min=0,
                max=100,
                value=10,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Percentage Bad Pixels</b><br>Percentage of bad pixels allowed."
            )
        else:
            raise ValueError(f"Invalid setting name: {setting_name}")

        return widget, instructions

    def get_settings(self) -> dict:
        for setting_name, widget in self.settings_widgets.items():
            self.settings[setting_name] = widget.value

        return self.settings.copy()

    def render(self) -> None:
        # Display the settings UI
        # Create the settings UI
        self.settings_ui = ipywidgets.Tab(
            children=[self.basic_settings_tab, self.advanced_settings_tab]
        )
        # self.settings_ui = ipywidgets.Accordion(
        #     children=[self.basic_settings_tab, self.advanced_settings_tab]
        # )
        self.settings_ui.set_title(0, "Basic Settings")
        self.settings_ui.set_title(1, "Advanced Settings")
        return self.settings_ui

    # def create_setting_widget(
    #     self, setting_name: str
    # ) -> Tuple[
    #     Union[ipywidgets.ToggleButton, ipywidgets.FloatSlider, ipywidgets.IntText],
    #     ipywidgets.HTML,
    # ]:
    #     # Create the widget for the setting
    #     if setting_name == "apply_cloud_mask":
    #         widget = ipywidgets.ToggleButton(
    #             description="Apply Cloud Mask", value=True, **self.action_style
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Apply Cloud Mask</b><br>Enable/disable cloud masking."
    #         )
    #     elif setting_name == "max_dist_ref":
    #         widget = ipywidgets.FloatSlider(
    #             description="Max Distance Reference",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Max Distance Reference</b><br>Maximum distance from the shoreline to search for reference points."
    #         )
    #     elif setting_name == "along_dist":
    #         widget = ipywidgets.FloatSlider(
    #             description="Alongshore Distance",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Alongshore Distance</b><br>Distance along the shoreline to search for reference points."
    #         )
    #     elif setting_name == "dist_clouds":
    #         widget = ipywidgets.FloatSlider(
    #             description="Distance to Clouds",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Distance to Clouds</b><br>Maximum distance from the shoreline to search for clouds."
    #         )
    #     elif setting_name == "min_beach_area":
    #         widget = ipywidgets.FloatSlider(
    #             description="Minimum Beach Area",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Minimum Beach Area</b><br>Minimum area of beach required to be considered a valid reference point."
    #         )
    #     elif setting_name == "min_length_sl":
    #         widget = ipywidgets.FloatSlider(
    #             description="Minimum Shoreline Length",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Minimum Shoreline Length</b><br>Minimum length of shoreline required to be considered a valid reference point."
    #         )
    #     elif setting_name == "cloud_thresh":
    #         widget = ipywidgets.FloatSlider(
    #             description="Cloud Threshold",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Cloud Threshold</b><br>Threshold for cloud detection."
    #         )
    #     elif setting_name == "min_points":
    #         widget = ipywidgets.IntText(
    #             description="Minimum Points", value=10, **self.action_style
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Minimum Points</b><br>Minimum number of reference points required to calculate shoreline."
    #         )
    #     elif setting_name == "max_std":
    #         widget = ipywidgets.FloatSlider(
    #             description="Maximum Standard Deviation",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Maximum Standard Deviation</b><br>Maximum standard deviation allowed for reference points."
    #         )
    #     elif setting_name == "max_range":
    #         widget = ipywidgets.FloatSlider(
    #             description="Maximum Range",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Maximum Range</b><br>Maximum range allowed for reference points."
    #         )
    #     elif setting_name == "min_chainage":
    #         widget = ipywidgets.FloatSlider(
    #             description="Minimum Chainage",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Minimum Chainage</b><br>Minimum chainage required to be considered a valid reference point."
    #         )
    #     elif setting_name == "multiple_inter":
    #         widget = ipywidgets.ToggleButton(
    #             description="Multiple Intersections", value=True, **self.action_style
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Multiple Intersections</b><br>Enable/disable multiple intersection detection."
    #         )
    #     elif setting_name == "prc_multiple":
    #         widget = ipywidgets.FloatSlider(
    #             description="Percentage Multiple Intersections",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Percentage Multiple Intersections</b><br>Percentage of multiple intersections allowed."
    #         )
    #     elif setting_name == "percent_no_data":
    #         widget = ipywidgets.FloatSlider(
    #             description="Percentage Bad Pixels",
    #             min=0,
    #             max=100,
    #             value=10,
    #             **self.action_style,
    #         )
    #         instructions = ipywidgets.HTML(
    #             value="<b>Percentage Bad Pixels</b><br>Percentage of bad pixels allowed."
    #         )
    #     else:
    #         raise ValueError(f"Invalid setting name: {setting_name}")

    #     return widget, instructions
