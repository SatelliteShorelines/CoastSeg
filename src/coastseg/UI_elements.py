# A module for common UI elements that can be used across different notebooks

# Standard Python imports
import os

# External Python imports
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipyfilechooser import FileChooser

class TidesSelector(widgets.VBox):
    def __init__(self):
        super().__init__()
        self.tide_model_selection = widgets.Dropdown(
            options=['FES2014', 'FES2022',],
            value='FES2022',
            description='Tide Model:',
            disabled=False,
            style={'description_width': 'initial'},
        )
        self.options = ['Use tide model', 'Upload tides file']

        self.tide_model_container,self.tide_model_selection = self.create_container_use_tide_model()
        self.file_chooser_container, self.file_chooser = self.create_file_chooser_with_clear(lambda x: print(x.selected), title='Select a CSV file of tides', filter_pattern='*.csv')
        
        self.slope_options = self.options
        self.radio_buttons = widgets.RadioButtons(options=self.slope_options, description='Select Tides', disabled=False)
        self.radio_buttons.observe(self.on_radio_change, names='value')
        
        self.output = widgets.Output()
        self.children = [self.radio_buttons, self.output]
        self.on_radio_change({'new': self.radio_buttons.value})  # Initialize with the default selection

    def on_radio_change(self, change):
        with self.output:
            clear_output(wait=True)
            if change['new'] == self.options[0]:
                display(self.tide_model_container)
            else:
                display(self.file_chooser_container)

    def create_container_use_tide_model(self ):
        instructions = widgets.HTML(
            value="""\
            <div>
                Select the tide model to calculate the tides from the dropdown list.<br>
                You must have the tide model installed on your system to use this option. Follow this guide on how to install the tide model:
                <a href="https://satelliteshorelines.github.io/CoastSeg/How-to-Download-Tide-Model/" target="_blank">Installation Guide</a>
            </div>
            """,
            layout=widgets.Layout(margin="0 0 0 0")
        )

        
        self.tide_model_selection = widgets.Dropdown(
            options=['FES2014', 'FES2022',],
            value='FES2022',
            description='Tide Model:',
            disabled=False,
            style={'description_width': 'initial'},
        )

        container = widgets.VBox([instructions, self.tide_model_selection], layout=widgets.Layout(width="100%"))
        return container,self.tide_model_selection

    def create_file_chooser_with_clear(self, callback, title="Select a file", filter_pattern="*.csv"):
        padding = "0px 0px 0px 5px"
        initial_path = os.getcwd()
        file_chooser = FileChooser(initial_path)
        file_chooser.filter_pattern = filter_pattern
        file_chooser.title = f"<b>{title}</b>"
        file_chooser.register_callback(callback)

        clear_button = widgets.Button(description="Clear", button_style="warning", layout=widgets.Layout(height="28px", padding=padding))
        clear_button.on_click(lambda b: file_chooser.reset())

        instructions = widgets.HTML(value="""Upload a CSV file containing the tides. Note any transescts that do not have tides will not be included in the tide correction.
                                    The CSV file must follow the file format listed here: 
                                     <a href="https://satelliteshorelines.github.io/CoastSeg/tide-file-format/" target="_blank" style="color: blue; text-decoration: underline;">View acceptable formats</a>""",
                            layout=widgets.Layout(margin="0 0 10px 0"))

        chooser = widgets.VBox([instructions, widgets.HBox([file_chooser, clear_button], layout=widgets.Layout(width="100%"))])
        return chooser, file_chooser

    @property
    def tides_file(self):
        # if the use tide model option is not selected return an empty string for the tides file
        if self.radio_buttons.value == self.options[0]:
            return ""
        else:
            if self.file_chooser.selected is None:
                return ""
            return self.file_chooser.selected

    @property
    def model(self):
        # if the use tide model option is selected return the selected tide model. Otherwise return an empty string
        if self.radio_buttons.value == self.options[0]:
            return self.tide_model_selection.value
        else:
            return ""


class BeachSlopeSelector(widgets.VBox):
    def __init__(self):
        super().__init__()
        self.beach_slope_text = widgets.FloatText(value=0.02, description="Beach Slope (m/m):", style={'description_width': 'initial'})
        self.file_chooser_container, self.file_chooser = self.create_file_chooser_with_clear(lambda x: print(x.selected), title='Select a CSV file of slopes', filter_pattern='*.csv')
        
        self.slope_options = ['Single slope', 'Upload slopes file']
        self.radio_buttons = widgets.RadioButtons(options=self.slope_options, description='Select Beach Slope Format', disabled=False)
        self.radio_buttons.observe(self.on_slope_change, names='value')
        
        self.output = widgets.Output()
        self.children = [self.radio_buttons, self.output]
        self.on_slope_change({'new': self.radio_buttons.value})  # Initialize with the default selection

    def on_slope_change(self, change):
        with self.output:
            clear_output(wait=True)
            if change['new'] == 'Single slope':
                display(self.beach_slope_text)
            else:
                display(self.file_chooser_container)

    def create_file_chooser_with_clear(self, callback, title="Select a file", filter_pattern="*.csv"):
        padding = "0px 0px 0px 5px"
        initial_path = os.getcwd()
        file_chooser = FileChooser(initial_path)
        file_chooser.filter_pattern = filter_pattern
        file_chooser.title = f"<b>{title}</b>"
        file_chooser.register_callback(callback)

        clear_button = widgets.Button(description="Clear", button_style="warning", layout=widgets.Layout(height="28px", padding=padding))
        clear_button.on_click(lambda b: file_chooser.reset())

        instructions = widgets.HTML(value="""Upload a CSV file containing the slopes. Note: any transescts that do not have a slope will have their slope set to the median slope value for the tide correction.
                                    The CSV file must follow the file format listed here: 
                                     <a href="https://satelliteshorelines.github.io/CoastSeg/slope-file-format/" target="_blank" style="color: blue; text-decoration: underline;">View acceptable formats</a>""",
                            layout=widgets.Layout(margin="0 0 10px 0"))

        chooser = widgets.VBox([instructions, widgets.HBox([file_chooser, clear_button], layout=widgets.Layout(width="100%"))])
        return chooser, file_chooser

    @property
    def value(self):
        if self.radio_buttons.value == 'Single slope':
            return self.beach_slope_text.value
        else:
            if self.file_chooser.selected is None:
                return ""
            return self.file_chooser.selected
