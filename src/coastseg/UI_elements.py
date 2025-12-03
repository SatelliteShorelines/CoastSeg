# A module for common UI elements that can be used across different notebooks

# Standard Python imports
import os
from typing import Callable, Tuple, Union

# External Python imports
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import clear_output, display


class TidesSelector(widgets.VBox):
    """UI widget for selecting tide data source and configuration."""

    def __init__(self) -> None:
        """
        Initializes tide selector widget with model and file upload options.

        Creates radio buttons for choosing between tide model and file upload,
        configures UI containers for each option.
        """
        super().__init__()
        self.tide_model_selection = widgets.Dropdown(
            options=[
                "FES2014",
                "FES2022",
            ],
            value="FES2022",
            description="Tide Model:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.options = ["Use tide model", "Upload tides file"]

        self.tide_model_container, self.tide_model_selection = (
            self.create_container_use_tide_model()
        )
        self.file_chooser_container, self.file_chooser = (
            self.create_file_chooser_with_clear(
                lambda x: print(x.selected),
                title="Select a CSV file of tides",
                filter_pattern="*.csv",
            )
        )

        self.slope_options = self.options
        self.radio_buttons = widgets.RadioButtons(
            options=self.slope_options, description="Select Tides", disabled=False
        )
        self.radio_buttons.observe(self.on_radio_change, names="value")

        self.output = widgets.Output()
        self.children = [self.radio_buttons, self.output]
        self.on_radio_change(
            {"new": self.radio_buttons.value}
        )  # Initialize with the default selection

    def on_radio_change(self, change):
        with self.output:
            clear_output(wait=True)
            if change["new"] == self.options[0]:
                display(self.tide_model_container)
            else:
                display(self.file_chooser_container)

    def create_container_use_tide_model(self) -> Tuple[widgets.VBox, widgets.Dropdown]:
        """
        Creates UI container for tide model selection.

        Returns:
            Tuple[widgets.VBox, widgets.Dropdown]: Container widget and dropdown for tide model selection.
        """
        instructions = widgets.HTML(
            value="""\
            <div>
                Select the tide model to calculate the tides from the dropdown list.<br>
                You must have the tide model installed on your system to use this option. Follow this guide on how to install the tide model:
                <a href="https://satelliteshorelines.github.io/CoastSeg/How-to-Download-Tide-Model/" target="_blank">Installation Guide</a>
            </div>
            """,
            layout=widgets.Layout(margin="0 0 0 0"),
        )

        self.tide_model_selection = widgets.Dropdown(
            options=[
                "FES2014",
                "FES2022",
            ],
            value="FES2022",
            description="Tide Model:",
            disabled=False,
            style={"description_width": "initial"},
        )

        container = widgets.VBox(
            [instructions, self.tide_model_selection],
            layout=widgets.Layout(width="100%"),
        )
        return container, self.tide_model_selection

    def create_file_chooser_with_clear(
        self,
        callback: Callable,
        title: str = "Select a file",
        filter_pattern: str = "*.csv",
    ) -> Tuple[widgets.VBox, FileChooser]:
        """
        Creates file chooser widget with clear button.

        Args:
            callback (Callable): Function to call when file is selected.
            title (str): Title for the file chooser.
            filter_pattern (str): File pattern filter.

        Returns:
            Tuple[widgets.VBox, FileChooser]: Container widget and file chooser instance.
        """
        padding = "0px 0px 0px 5px"
        initial_path = os.getcwd()
        file_chooser = FileChooser(initial_path)
        file_chooser.filter_pattern = filter_pattern
        file_chooser.title = f"<b>{title}</b>"
        file_chooser.register_callback(callback)

        clear_button = widgets.Button(
            description="Clear",
            button_style="warning",
            layout=widgets.Layout(height="28px", padding=padding),
        )
        clear_button.on_click(lambda b: file_chooser.reset())

        instructions = widgets.HTML(
            value="""Upload a CSV file containing the tides. Note any transescts that do not have tides will not be included in the tide correction.
                                    The CSV file must follow the file format listed here: 
                                     <a href="https://satelliteshorelines.github.io/CoastSeg/tide-file-format/" target="_blank" style="color: blue; text-decoration: underline;">View acceptable formats</a>""",
            layout=widgets.Layout(margin="0 0 10px 0"),
        )

        chooser = widgets.VBox(
            [
                instructions,
                widgets.HBox(
                    [file_chooser, clear_button], layout=widgets.Layout(width="100%")
                ),
            ]
        )
        return chooser, file_chooser

    @property
    def tides_file(self) -> str:
        """
        Gets the selected tides file path.

        Returns:
            str: Path to selected CSV file or empty string if using tide model.
        """
        # if the use tide model option is not selected return an empty string for the tides file
        if self.radio_buttons.value == self.options[0]:
            return ""
        else:
            if self.file_chooser.selected is None:
                return ""
            return self.file_chooser.selected

    @property
    def model(self) -> str:
        """
        Gets the selected tide model name.

        Returns:
            str: Tide model name or empty string if using file upload.
        """
        # if the use tide model option is selected return the selected tide model. Otherwise return an empty string
        if self.radio_buttons.value == self.options[0]:
            return self.tide_model_selection.value
        else:
            return ""


class BeachSlopeSelector(widgets.VBox):
    """UI widget for selecting beach slope configuration."""

    def __init__(self) -> None:
        """
        Initializes beach slope selector with single value and file upload options.

        Creates radio buttons for choosing between single slope value and file upload,
        configures UI containers for each option.
        """
        super().__init__()
        self.beach_slope_text = widgets.FloatText(
            value=0.02,
            description="Beach Slope (m/m):",
            style={"description_width": "initial"},
        )
        self.file_chooser_container, self.file_chooser = (
            self.create_file_chooser_with_clear(
                lambda x: print(x.selected),
                title="Select a CSV file of slopes",
                filter_pattern="*.csv",
            )
        )

        self.slope_options = ["Single slope", "Upload slopes file"]
        self.radio_buttons = widgets.RadioButtons(
            options=self.slope_options,
            description="Select Beach Slope Format",
            disabled=False,
        )
        self.radio_buttons.observe(self.on_slope_change, names="value")

        self.output = widgets.Output()
        self.children = [self.radio_buttons, self.output]
        self.on_slope_change(
            {"new": self.radio_buttons.value}
        )  # Initialize with the default selection

    def create_file_chooser_with_clear(
        self, callback, title="Select a file", filter_pattern="*.csv"
    ):
        padding = "0px 0px 0px 5px"
        initial_path = os.getcwd()
        file_chooser = FileChooser(initial_path)
        file_chooser.filter_pattern = filter_pattern
        file_chooser.title = f"<b>{title}</b>"
        file_chooser.register_callback(callback)

        clear_button = widgets.Button(
            description="Clear",
            button_style="warning",
            layout=widgets.Layout(height="28px", padding=padding),
        )
        clear_button.on_click(lambda b: file_chooser.reset())

        instructions = widgets.HTML(
            value="""Upload a CSV file containing the slopes. Note: any transescts that do not have a slope will have their slope set to the median slope value for the tide correction.
                                    The CSV file must follow the file format listed here: 
                                     <a href="https://satelliteshorelines.github.io/CoastSeg/slope-file-format/" target="_blank" style="color: blue; text-decoration: underline;">View acceptable formats</a>""",
            layout=widgets.Layout(margin="0 0 10px 0"),
        )

        chooser = widgets.VBox(
            [
                instructions,
                widgets.HBox(
                    [file_chooser, clear_button], layout=widgets.Layout(width="100%")
                ),
            ]
        )
        return chooser, file_chooser

    def on_slope_change(self, change):
        with self.output:
            clear_output(wait=True)
            if change["new"] == "Single slope":
                display(self.beach_slope_text)
            else:
                display(self.file_chooser_container)

    @property
    def value(self) -> Union[float, str]:
        """
        Gets the selected beach slope value or file path.

        Returns:
            Union[float, str]: Slope value if single slope selected,
                             file path if file upload selected, or empty string if no file.
        """
        if self.radio_buttons.value == "Single slope":
            return self.beach_slope_text.value
        else:
            if self.file_chooser.selected is None:
                return ""
            return self.file_chooser.selected
