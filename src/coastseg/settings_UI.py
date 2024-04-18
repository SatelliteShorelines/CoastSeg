# standard python imports
# external python imports
import ipywidgets
import datetime
from typing import List, Union, Optional, Tuple
from ipywidgets import Layout, Box,VBox

GRID_LAYOUT = Layout(
    display='grid',
    width='100%',  # Use 100% of the container width
    grid_template_columns='repeat(6, 90px)',  # Create 6 flexible columns
    grid_template_rows='repeat(2, auto)',  # Create 2 rows, size to content
    grid_gap='5px',  # Adjust the gap as needed
    overflow='auto'  # Allow scrollbar if content overflows
)
checkbox_layout = Layout(width='auto')  # This sets the width of each checkbox to be only as wide as necessary


class ButtonColors:
    REMOVE = "red"
    LOAD = "#69add1"
    ACTION = "#ae3cf0"
    SAVE = "#50bf8f"
    CLEAR = "#a3adac"


def str_to_bool(var: str) -> bool:
    return var.lower().strip() == "true"


def convert_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date: {date_str}. Expected format: 'YYYY-MM-DD'.{e}")

class CustomMonthSelector(VBox):
    month_to_num = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }
    def __init__(self, checkboxes, layout):
        super().__init__(children=checkboxes, layout=layout)
        # Observe changes in each checkbox and update the value property accordingly
        for checkbox in checkboxes:
            checkbox.observe(self._update_value, names='value')
        self._value = [CustomMonthSelector.month_to_num[checkbox.description] for checkbox in self.children if checkbox.value]
    
    @property
    def value(self):
        return self._value
    
    def _update_value(self, change):
        self._value = [CustomMonthSelector.month_to_num[checkbox.description] for checkbox in self.children if checkbox.value]

    @value.setter
    def value(self, values):
        # set all the checkboxes to False
        for checkbox in self.children:
            checkbox.value = False
        # set the checkboxes to True for the values in the list
        for val in values:
            self.children[val-1].value = True 
                
        self._value = [CustomMonthSelector.month_to_num[checkbox.description] for checkbox in self.children if checkbox.value]


class DateBox(ipywidgets.HBox):
    def __init__(self, start_date=None, end_date=None, **kwargs):
        if start_date is None:
            start_date = datetime.date(2018, 12, 1)
        if end_date is None:
            end_date = datetime.date(2019, 3, 1)

        self.start_date = ipywidgets.DatePicker(
            description="Start Date",
            value=start_date,
            disabled=False,
        )
        self.end_date = ipywidgets.DatePicker(
            description="End Date",
            value=end_date,
            disabled=False,
        )
        super().__init__([self.start_date, self.end_date], **kwargs)

    @property
    def value(self):
        return [str(self.start_date.value), str(self.end_date.value)]

    @value.setter
    def value(self, values):
        if len(values) != 2:
            raise ValueError("You must provide a list of two dates.")

        start_date, end_date = values

        if isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.date.fromisoformat(end_date)

        self.start_date.value = start_date
        self.end_date.value = end_date

    @property
    def options(self):
        return [self.start_date.value, self.end_date.value]

    @options.setter
    def options(self, values):
        if len(values) != 2:
            raise ValueError("You must provide a list of two dates.")

        start_date, end_date = values

        if isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.date.fromisoformat(end_date)

        self.start_date.value = start_date
        self.end_date.value = end_date


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
                "drop_intersection_pts",
            ]
        if advanced_settings is None:
            advanced_settings = [
                "along_dist",
                "min_points",
                "max_std",
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
        """
        Create a settings tab with widgets for each setting.
        
        Places the cloud_thresh setting at the second position in the tab.

        Args:
            settings (List[str]): A list of setting names.

        Returns:
            ipywidgets.VBox: The settings tab containing the widgets for each setting.
        """
        # Create the settings tab
        tab_contents = []
        for setting_name in settings:
            # Create the widget for the setting
            widget, instructions = self.create_setting_widget(setting_name)
            
            # Add the widget and instructions to the tab contents
            if setting_name == 'cloud_thresh':
                tab_contents.insert(2, ipywidgets.VBox([instructions, widget]))
            else:
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
        if title is None:
            raise ValueError("Title cannot be None.")
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
        """
        Create a setting widget based on the given setting name.

        Parameters:
            setting_name (str): The name of the setting.

        Returns:
            Tuple[Union[ipywidgets.ToggleButton, ipywidgets.FloatSlider, ipywidgets.IntText], ipywidgets.HTML]:
                A tuple containing the widget for the setting and the corresponding instructions HTML widget.

        Raises:
            ValueError: If the setting name is invalid.
        """
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
                description="Reference Shoreline Buffer",
                value=100,
                min=5,
                max=1000,
                step=1,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Reference Shoreline Buffer</b><br>Max size of the buffer around reference shorelines in which shorelines can be extracted"
            )
        elif setting_name == "along_dist":
            widget = ipywidgets.IntSlider(
                description="Alongshore Distance",
                value=25,
                min=10,
                max=100,
                step=1,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Alongshore Distance</b><br> Along-shore distance over which to consider shoreline points to compute median intersection with transects."
            )
        elif setting_name == "dist_clouds":
            widget = ipywidgets.IntSlider(
                description="Distance to Clouds",
                value=300,
                min=0,
                max=1000,
                step=1,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Allowed Distance from Clouds</b><br>Any shorelines within this distance of a cloud will be removed."
            )
        elif setting_name == "min_beach_area":
            widget = ipywidgets.IntSlider(
                description="Minimum Beach Area",
                min=10,
                max=10000,
                value=1000,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Beach Area</b><br>Minimum area (sqm) for object to be labeled as beach"
            )
        elif setting_name == "min_length_sl":
            widget = ipywidgets.IntSlider(
                description="Minimum Shoreline Length",
                value=500,
                min=50,
                max=1000,
                step=1,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Shoreline Length</b><br>Minimum length of shoreline in meters."
            )
        elif setting_name == "cloud_thresh":
            widget = ipywidgets.FloatSlider(
                description="Cloud Threshold",
                value=0.8,
                min=0,
                max=1,
                step=0.01,
                readout_format=".2f",
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Cloud Threshold</b><br>Maximum percentage of cloud pixels in an image."
            )
        elif setting_name == "min_points":
            widget = ipywidgets.BoundedIntText(
                description="Minimum Points",
                value=3,
                min=1,
                max=100,
                step=1,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Points</b><br>Minimum number of shoreline points required to calculate shoreline."
            )
        elif setting_name == "max_std":
            widget = ipywidgets.BoundedFloatText(
                description="Maximum Standard Deviation",
                value=15.0,
                min=1.0,
                max=100.0,
                step=1.0,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Maximum Standard Deviation</b><br>Maximum standard deviation allowed for reference points."
            )
        elif setting_name == "max_range":
            widget = ipywidgets.BoundedFloatText(
                description="Maximum Range(m)",
                min=1.0,
                max=100.0,
                value=30.0,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Maximum Range</b><br>Max range for shoreline points within the alongshore range, if range is above this value a NaN is returned for this intersection."
            )
        elif setting_name == "min_chainage":
            widget = ipywidgets.BoundedFloatText(
                description="Minimum Chainage",
                value=-100.0,
                min=-500.0,
                max=-1.0,
                step=-1.0,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Minimum Chainage</b><br>Max distance landward of the transect origin that an intersection is accepted, beyond this point a NaN is returned."
            )
        elif setting_name == "multiple_inter":
            widget = ipywidgets.Select(
                description="Multiple Intersections",
                options=["auto", "nan", "max"],
                value="auto",
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Outliers Mode</b><br>Enable/disable multiple intersection detection."
            )
        elif setting_name == "prc_multiple":
            widget = ipywidgets.BoundedFloatText(
                description="Percentage Multiple Intersections",
                value=0.1,
                min=0.0,
                max=1.0,
                step=0.01,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Percentage Multiple Intersections</b><br>Percentage of points whose std > max_std that will be set to 'max'.Only in 'auto' mode."
            )
        elif setting_name == "percent_no_data":
            widget = ipywidgets.FloatSlider(
                description="% Bad Pixels",
                value=0.80,
                min=0.0,
                max=1.0,
                step=0.01,
                style={"description_width": "initial"},
            )
            instructions = ipywidgets.HTML(
                value="<b>Percentage Bad Pixels</b><br>Max percentage of bad pixels allowed in the requested images. Bad pixels are no data or cloud pixels."
            )
        elif setting_name == "drop_intersection_pts":
            widget = ipywidgets.Checkbox(
                value=False,
                description="Drop intersection points not on transects",
                indent=False,  # To align the description with the label
            )
            instructions = ipywidgets.HTML(
                value="<b>Drop Intersection Points Not On Transect</b><br>Activate to filter out shoreline intersection points that are not located on the transect they were detected on"
            )
        elif setting_name == "dates":
            widget = DateBox()
            instructions = ipywidgets.HTML(
                value="<b>Pick a date:</b>",
            )
        elif setting_name == "months_list":
            # Create a list of checkboxes for each month
            months = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
            checkboxes = [ipywidgets.Checkbox(description=month, value=True, indent=False,layout=checkbox_layout,) for month in months]
            widget = CustomMonthSelector(checkboxes, GRID_LAYOUT)
            instructions = ipywidgets.HTML(
                value="<b>(Optional) Choose months within the date range to download imagery within</b>",
            )            
            
        else:
            raise ValueError(f"Invalid setting name: {setting_name}")

        return widget, instructions

    def set_settings(self, settings: dict) -> None:
        """
        Set the settings of the UI widgets based on the provided dictionary.

        Args:
            settings (dict): A dictionary containing the settings to be applied.

        Returns:
            None
        """
        for setting_name, widget in self.settings_widgets.items():
            if setting_name in settings:
                if isinstance(widget, DateBox):
                    widget.value = list(map(convert_date, settings[setting_name]))
                elif isinstance(widget.value, str):
                    widget.value = str(settings[setting_name])
                elif isinstance(widget.value, bool):
                    if isinstance(settings[setting_name], str):
                        widget.value = str_to_bool(settings[setting_name])
                    else:
                        widget.value = bool(settings[setting_name])
                else:
                    widget.value = settings[setting_name]

    def get_settings(self) -> dict:
        """
        Retrieves the current settings from the settings widgets and returns them as a dictionary.
        
        For certain settings, the value is converted to the appropriate type before being added to the dictionary.

        Returns:
            dict: A dictionary containing the current settings.
        """
        for setting_name, widget in self.settings_widgets.items():
            self.settings[setting_name] = widget.value

        if "sat_list" in self.settings:
            sat_tuple = self.settings["sat_list"]
            self.settings["sat_list"] = list(sat_tuple)

        if "apply_cloud_mask" in self.settings:
            apply_cloud_mask = self.settings["apply_cloud_mask"]
            self.settings["apply_cloud_mask"] = str_to_bool(apply_cloud_mask)

        if "cloud_mask_issue" in self.settings:
            cloud_mask_issue = self.settings["cloud_mask_issue"]
            self.settings["cloud_mask_issue"] = str_to_bool(cloud_mask_issue)

        return self.settings.copy()

    def render(self) -> None:
        """
        Renders the settings UI.

        This method displays the settings UI and returns the UI widget.

        Returns:
            ipywidgets.Accordion: The settings UI widget.
        """
        # Display the settings UI
        # Create the settings UI
        # Define a layout with a maximum height
        layout = Layout(max_height='320px', overflow='auto')
        settings_tabs = ipywidgets.Tab(
            children=[self.basic_settings_tab, self.advanced_settings_tab]
        )
        settings_tabs.set_title(0, "Basic Settings")
        settings_tabs.set_title(1, "Advanced Settings")
        # Wrap each child widget in a Box with the defined layout
        children = [Box([settings_tabs], layout=layout)]
        self.settings_ui = ipywidgets.Accordion(children=children, selected_index=0)
        self.settings_ui.set_title(0, "Settings")
        return self.settings_ui
