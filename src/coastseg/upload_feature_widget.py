from typing import Set
import ipywidgets as widgets
from ipywidgets import VBox
from IPython.display import display
from ipyfilechooser import FileChooser

"""
This code defines a FileUploader class, which provides a GUI for uploading GeoJSON files.
It uses IPython widgets to create interactive elements like dropdowns, buttons, and file choosers.
"""


class FileUploader:
    def __init__(
        self,
        dropdown_options: Set[str] = set(),
        filter_pattern: str = None,
        file_selection_title: str = "",
        starting_directory: str = "",
    ):
        """
        Initializes the FileUploader class.

        Parameters:
        - dropdown_options (Set[str]): Set of dropdown options for file types. Default is an empty set.
        - filter_pattern (str): File filter pattern for the file dialog. Default is None.
        - file_selection_title (str): Title for the file selection dialog. Default is an empty string.
        - starting_directory (str): Starting directory for the file dialog. Default is an empty string.
        """
        # Initialize widget elements for file uploading
        self.filenames = widgets.SelectMultiple(options=[])
        self.remove_button = widgets.Button(
            description="x",
            button_style="danger",
            layout=widgets.Layout(width="50px", height="28px"),
        )
        uploaded_files_title = widgets.HTML(value="<b>Uploaded Files</b>")
        self.remove_widget = widgets.VBox(
            [uploaded_files_title, widgets.HBox([self.filenames, self.remove_button])]
        )
        self.title = widgets.HTML(value="<h1>Upload a GeoJSON File</h1>")

        # Create a dropdown with specified options, or default to ['transects', 'shorelines']
        self.dropdown = widgets.Dropdown(options=list(dropdown_options))
        if not dropdown_options:
            self.dropdown = widgets.Dropdown(options=["transects", "shorelines"])

        self.files_dict = {}
        self.file_dialog = FileChooser(starting_directory)
        self.file_dialog_row = widgets.HBox([])
        self.file_dialog_row.children = [self.file_dialog]

        # Setup event handlers
        self.remove_button.on_click(self.remove_file)

        # Initialize the file_dialog with specified options
        self.file_dialog.title = f"<b>{file_selection_title}</b>"
        self.file_dialog.register_callback(self.save_file)
        if not isinstance(filter_pattern, list):
            filter_pattern = [filter_pattern]
        self.file_dialog.filter_pattern = filter_pattern

    # Remove the selected file (if any) from the files_dict and update filenames options
    def remove_file(self, button: widgets.Button):
        """
        Removes the selected file from the files_dict and updates the filenames options.

        Parameters:
        - button(widgets.Button): The button widget that triggered the event.
        """
        selected_files = self.filenames.value
        if selected_files:
            for selected_file in selected_files:
                for key in self.files_dict:
                    if self.files_dict[key] == selected_file:
                        del self.files_dict[key]
                        break
            self.filenames.options = list(self.files_dict.values())

    # Save the selected file in the files_dict and update filenames options
    def save_file(self, selected: tuple):
        """
        Saves the selected file in the files_dict and updates the filenames options.

        Parameters:
        - selected(tuple): The selected file from the file dialog.
        """
        feature = self.dropdown.value
        self.files_dict[feature] = self.file_dialog.selected
        self.filenames.options = list(self.files_dict.values())

    # Display the FileUploader interface
    def display(self):
        """
        Displays the FileUploader interface.
        """
        display(self.title, self.dropdown, self.remove_widget, self.file_dialog_row)

    # Display the FileUploader interface
    def get_FileUploader_widget(self):
        """
        Displays the FileUploader interface.
        """
        return VBox([self.title, self.dropdown, self.remove_widget, self.file_dialog_row])