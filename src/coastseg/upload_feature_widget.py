from typing import Any, Collection, Dict, List, Union

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display
from ipywidgets import Box, VBox

"""
This code defines a FileUploader class, which provides a GUI for uploading GeoJSON files.
It uses IPython widgets to create interactive elements like dropdowns, buttons, and file choosers.
"""


class FileUploader:
    """Widget for uploading and managing GeoJSON files with interactive UI components."""

    def __init__(
        self,
        title: str = "Upload a GeoJSON File",
        instructions: str = "",
        dropdown_options: Collection[str] = set(),
        filter_pattern: str = "*.geojson",
        file_selection_title: str = "",
        starting_directory: str = "",
        max_width: int = 400,
    ) -> None:
        """
        Initializes file uploader widget with customizable options.

        Args:
            title (str): Widget title text.
            instructions (str): Instructional text for users.
            dropdown_options (Set[str]): Available feature type options.
            filter_pattern (str): File filter pattern for file selection.
            file_selection_title (str): Title for file selection dialog.
            starting_directory (str): Initial directory for file chooser.
            max_width (int): Maximum widget width in pixels.
        """
        self.filenames = widgets.SelectMultiple(options=[])
        self.remove_button = widgets.Button(
            description="Remove",
            button_style="danger",
            layout=widgets.Layout(width="75px", height="28px"),
        )
        uploaded_files_title = widgets.HTML(value="<b>Uploaded Files</b>")
        self.remove_widget = widgets.VBox(
            [uploaded_files_title, widgets.HBox([self.filenames, self.remove_button])]
        )

        # Convert max_width to string form
        self.max_width = f"{max_width}px"

        # If title or instructions are empty, don't create the widget.
        self.title = widgets.HTML(value=f"<h2>{title}</h2>") if title else None
        self.instructions = (
            Box(
                [
                    widgets.HTML(
                        value=instructions,
                        layout=widgets.Layout(
                            overflow="auto",
                            white_space="pre-wrap",
                            max_width=self.max_width,
                        ),
                    )
                ],
                layout=widgets.Layout(overflow="auto", width="auto"),
            )
            if instructions
            else None
        )

        self.dropdown = widgets.Dropdown(
            options=(
                list(dropdown_options)
                if dropdown_options
                else ["transects", "shorelines", "shoreline extraction area"]
            )
        )

        self.files_dict: Dict[str, str] = {}
        self.file_dialog = FileChooser(starting_directory)
        self.file_dialog_row = widgets.HBox([self.file_dialog])

        self._initialize_file_dialog(filter_pattern, file_selection_title)

        # Register event handlers
        self.remove_button.on_click(self.remove_file)

    def _initialize_file_dialog(
        self, filter_pattern: Union[str, List[str]], file_selection_title: str
    ) -> None:
        """
        Configures file dialog with filter and callback.

        Args:
            filter_pattern (Union[str, List[str]]): File pattern filter(s).
            file_selection_title (str): Title for file selection dialog.
        """
        self.file_dialog.title = f"<b>{file_selection_title}</b>"
        self.file_dialog.filter_pattern = (
            filter_pattern if isinstance(filter_pattern, list) else [filter_pattern]
        )
        self.file_dialog.register_callback(self.save_file)

    def remove_file(self, button: widgets.Button) -> None:
        """
        Removes selected files from the uploaded files list.

        Args:
            button (widgets.Button): Button widget that triggered the event.
        """
        selected_files = self.filenames.value
        for selected_file in selected_files:
            self.files_dict = {
                k: v for k, v in self.files_dict.items() if v != selected_file
            }
        self.filenames.options = list(self.files_dict.values())

    def save_file(self, selected: Any) -> None:
        """
        Saves selected file to the files dictionary.

        Args:
            selected (Any): Selected file information from file chooser.
        """
        feature = self.dropdown.value
        self.files_dict[feature] = self.file_dialog.selected
        self.filenames.options = list(self.files_dict.values())
        # Clear the file selection
        self.file_dialog.reset()

    def _get_widgets_to_display(self) -> List[widgets.Widget]:
        """
        Gets ordered list of widgets for display.

        Returns:
            List[widgets.Widget]: Non-null widgets in display order.
        """
        # Order of widgets
        return [
            w
            for w in [
                self.title,
                self.instructions,
                self.dropdown,
                self.remove_widget,
                self.file_dialog_row,
            ]
            if w is not None
        ]

    def display(self) -> None:
        """Displays all file uploader widgets in the current output."""
        display(*self._get_widgets_to_display())

    def get_FileUploader_widget(self) -> VBox:
        """
        Gets file uploader widget as a VBox container.

        Returns:
            VBox: Container widget with all file uploader components.
        """
        return VBox(
            self._get_widgets_to_display(),
            layout=widgets.Layout(max_width=self.max_width),
        )
