import pytest
from coastseg.settings_UI import Settings_UI
import ipywidgets


@pytest.fixture
def settings_dashboard():
    basic_settings = [
        "dates",
        "max_dist_ref",
        "min_length_sl",
        "min_beach_area",
        "dist_clouds",
        "apply_cloud_mask",
        "cloud_thresh",
        "percent_no_data",
    ]

    settings_dashboard = Settings_UI(basic_settings)
    return settings_dashboard


def test_set_settings_with_datebox(settings_dashboard):
    settings = {
        "dates": ["2022-01-01", "2022-01-02"],
        "max_dist_ref": 30,
        "bogus_settings": True,  # this settings is not in the basic settings
        "cloud_thresh": 0.8,
    }
    settings_dashboard.set_settings(settings)
    assert settings_dashboard.settings_widgets["dates"].value == [
        "2022-01-01",
        "2022-01-02",
    ]
    assert settings_dashboard.settings_widgets["max_dist_ref"].value == 30
    assert settings_dashboard.settings_widgets["cloud_thresh"].value == 0.8
    assert "bogus_settings" not in settings_dashboard.settings_widgets.keys()
    # assert settings_ui.settings_widgets["bogus_settings"].value is True


def test_set_settings_with_string(settings_dashboard):
    settings = {
        "dates": ["2022-01-01", "2022-01-02"],
        "max_dist_ref": 30,
        "apply_cloud_mask": True,  # this settings should be converted to a str to be rendered in the UI
        "cloud_thresh": 0.8,
    }
    settings_dashboard.set_settings(settings)
    assert settings_dashboard.settings_widgets["dates"].value == [
        "2022-01-01",
        "2022-01-02",
    ]
    assert settings_dashboard.settings_widgets["max_dist_ref"].value == 30
    assert settings_dashboard.settings_widgets["cloud_thresh"].value == 0.8
    assert settings_dashboard.settings_widgets["apply_cloud_mask"].value == "True"


def test_set_settings_with_bool(settings_dashboard):
    settings = {
        "dates": ["2022-01-01", "2022-01-02"],
        "max_dist_ref": 30,
        "apply_cloud_mask": True,  # this settings should be converted to a str to be rendered in the UI
        "cloud_thresh": 0.8,
        "image_size_filter": "False",
    }
    # add a custom widget which only accepts bools
    image_size_filter_checkbox = ipywidgets.Checkbox(
        value=True,
        description="Enable Image Size Filter",
        indent=False,  # To align the description with the label
    )
    settings_dashboard.add_custom_widget(
        image_size_filter_checkbox,
        "image_size_filter",
        "Image Size Filter",
        "Activate to filter out images that are smaller than 60% of the Region of Interest (ROI).",
        advanced=False,
        index=-1,
    )
    settings_dashboard.set_settings(settings)

    assert "image_size_filter" in settings_dashboard.settings_widgets.keys()
    assert settings_dashboard.settings_widgets["dates"].value == [
        "2022-01-01",
        "2022-01-02",
    ]
    assert settings_dashboard.settings_widgets["max_dist_ref"].value == 30
    assert settings_dashboard.settings_widgets["cloud_thresh"].value == 0.8
    assert settings_dashboard.settings_widgets["apply_cloud_mask"].value == "True"
    assert settings_dashboard.settings_widgets["image_size_filter"].value == False


def test_add_custom_widget(settings_dashboard):
    settings = {
        "dates": ["2022-01-01", "2022-01-02"],
        "max_dist_ref": 30,
        "apply_cloud_mask": True,  # this settings should be converted to a str to be rendered in the UI
        "cloud_thresh": 0.8,
        "image_size_filter": "False",
    }

    instructions = (
        "Sand color on beach for model to detect 'dark' (grey/black) 'bright' (white)"
    )
    sand_widget = ipywidgets.Dropdown(
        options=["default", "latest", "dark", "bright"],
        value="default",
        description="sand_color :",
        disabled=False,
    )

    settings_dashboard.add_custom_widget(
        sand_widget,
        "sand_color",
        "Select Sand Color",
        instructions,
        advanced=True,
        index=0,
    )
    satellite_selection = ipywidgets.SelectMultiple(
        options=["L5", "L7", "L8", "L9", "S2"],
        value=["L8"],
        description="Satellites",
        disabled=False,
    )
    cloud_mask_issue = ipywidgets.ToggleButtons(
        options=["False", "True"],
        description=" Switch to True if sand pixels are masked (in black) on many images",
        disabled=False,
        button_style="",
        tooltips=[
            "No cloud mask issue",
            "Fix cloud masking",
        ],
    )
    settings_dashboard.add_custom_widget(
        cloud_mask_issue,
        "cloud_mask_issue",
        "Cloud Mask Issue",
        "Switch to True if sand pixels are masked (in black) on many images",
        advanced=True,
        index=-1,
    )

    settings_dashboard.add_custom_widget(
        satellite_selection,
        "sat_list",
        "Select Satellites",
        "Pick multiple satellites by holding the control key",
        advanced=False,
        index=1,
    )

    # add a custom widget which only accepts bools
    image_size_filter_checkbox = ipywidgets.Checkbox(
        value=True,
        description="Enable Image Size Filter",
        indent=False,  # To align the description with the label
    )
    settings_dashboard.add_custom_widget(
        image_size_filter_checkbox,
        "image_size_filter",
        "Image Size Filter",
        "Activate to filter out images that are smaller than 60% of the Region of Interest (ROI).",
        advanced=False,
        index=-1,
    )

    assert "image_size_filter" in settings_dashboard.settings_widgets.keys()
    assert "sat_list" in settings_dashboard.settings_widgets.keys()
    assert "cloud_mask_issue" in settings_dashboard.settings_widgets.keys()
    assert "sand_color" in settings_dashboard.settings_widgets.keys()


def test_add_custom_widget_set_custom_settings(settings_dashboard):
    settings = {
        "dates": ["2022-01-01", "2022-01-02"],
        "max_dist_ref": 30,
        "apply_cloud_mask": True,  # custom widget added with add_custom_widget
        "cloud_thresh": 0.8,
        "image_size_filter": "False",
        "sand_color": "dark",  # custom widget added with add_custom_widget
        "sat_list": ["L9", "S2"],  # custom widget added with add_custom_widget
        "cloud_mask_issue": "True",  # custom widget added with add_custom_widget
    }

    instructions = (
        "Sand color on beach for model to detect 'dark' (grey/black) 'bright' (white)"
    )
    sand_widget = ipywidgets.Dropdown(
        options=["default", "latest", "dark", "bright"],
        value="default",
        description="sand_color :",
        disabled=False,
    )

    settings_dashboard.add_custom_widget(
        sand_widget,
        "sand_color",
        "Select Sand Color",
        instructions,
        advanced=True,
        index=0,
    )
    satellite_selection = ipywidgets.SelectMultiple(
        options=["L5", "L7", "L8", "L9", "S2"],
        value=["L8"],
        description="Satellites",
        disabled=False,
    )
    cloud_mask_issue = ipywidgets.ToggleButtons(
        options=["False", "True"],
        description=" Switch to True if sand pixels are masked (in black) on many images",
        disabled=False,
        button_style="",
        tooltips=[
            "No cloud mask issue",
            "Fix cloud masking",
        ],
    )
    settings_dashboard.add_custom_widget(
        cloud_mask_issue,
        "cloud_mask_issue",
        "Cloud Mask Issue",
        "Switch to True if sand pixels are masked (in black) on many images",
        advanced=True,
        index=-1,
    )

    settings_dashboard.add_custom_widget(
        satellite_selection,
        "sat_list",
        "Select Satellites",
        "Pick multiple satellites by holding the control key",
        advanced=False,
        index=1,
    )

    # add a custom widget which only accepts bools
    image_size_filter_checkbox = ipywidgets.Checkbox(
        value=True,
        description="Enable Image Size Filter",
        indent=False,  # To align the description with the label
    )
    settings_dashboard.add_custom_widget(
        image_size_filter_checkbox,
        "image_size_filter",
        "Image Size Filter",
        "Activate to filter out images that are smaller than 60% of the Region of Interest (ROI).",
        advanced=False,
        index=-1,
    )
    settings_dashboard.set_settings(settings)

    assert "image_size_filter" in settings_dashboard.settings_widgets.keys()
    assert "sat_list" in settings_dashboard.settings_widgets.keys()
    assert "cloud_mask_issue" in settings_dashboard.settings_widgets.keys()
    assert "sand_color" in settings_dashboard.settings_widgets.keys()
    assert settings_dashboard.settings_widgets["image_size_filter"].value == False
    assert settings_dashboard.settings_widgets["sat_list"].value == ("L9", "S2")
    assert settings_dashboard.settings_widgets["cloud_mask_issue"].value == "True"
    assert settings_dashboard.settings_widgets["sand_color"].value == "dark"


def test_get_settings_custom_widgets(settings_dashboard):
    settings = {
        "dates": ["2022-01-01", "2022-01-02"],
        "max_dist_ref": 30,
        "apply_cloud_mask": True,  # custom widget added with add_custom_widget
        "cloud_thresh": 0.8,
        "percent_no_data": 0.8,
        "image_size_filter": "False",
        "sand_color": "dark",  # custom widget added with add_custom_widget
        "sat_list": ["L9", "S2"],  # custom widget added with add_custom_widget
        "cloud_mask_issue": "True",  # custom widget added with add_custom_widget
    }

    instructions = (
        "Sand color on beach for model to detect 'dark' (grey/black) 'bright' (white)"
    )
    sand_widget = ipywidgets.Dropdown(
        options=["default", "latest", "dark", "bright"],
        value="default",
        description="sand_color :",
        disabled=False,
    )

    settings_dashboard.add_custom_widget(
        sand_widget,
        "sand_color",
        "Select Sand Color",
        instructions,
        advanced=True,
        index=0,
    )
    satellite_selection = ipywidgets.SelectMultiple(
        options=["L5", "L7", "L8", "L9", "S2"],
        value=["L8"],
        description="Satellites",
        disabled=False,
    )
    cloud_mask_issue = ipywidgets.ToggleButtons(
        options=["False", "True"],
        description=" Switch to True if sand pixels are masked (in black) on many images",
        disabled=False,
        button_style="",
        tooltips=[
            "No cloud mask issue",
            "Fix cloud masking",
        ],
    )
    settings_dashboard.add_custom_widget(
        cloud_mask_issue,
        "cloud_mask_issue",
        "Cloud Mask Issue",
        "Switch to True if sand pixels are masked (in black) on many images",
        advanced=True,
        index=-1,
    )

    settings_dashboard.add_custom_widget(
        satellite_selection,
        "sat_list",
        "Select Satellites",
        "Pick multiple satellites by holding the control key",
        advanced=False,
        index=1,
    )

    # add a custom widget which only accepts bools
    image_size_filter_checkbox = ipywidgets.Checkbox(
        value=True,
        description="Enable Image Size Filter",
        indent=False,  # To align the description with the label
    )
    settings_dashboard.add_custom_widget(
        image_size_filter_checkbox,
        "image_size_filter",
        "Image Size Filter",
        "Activate to filter out images that are smaller than 60% of the Region of Interest (ROI).",
        advanced=False,
        index=-1,
    )
    settings_dashboard.set_settings(settings)

    expected_settings = {
        "dates": ["2022-01-01", "2022-01-02"],
        "max_dist_ref": 30,
        "min_length_sl": 500,
        "min_beach_area": 1000,
        "dist_clouds": 300,
        "apply_cloud_mask": True,
        "cloud_thresh": 0.8,
        "percent_no_data": 0.8,
        "along_dist": 25,
        "min_points": 3,
        "max_std": 15.0,
        "max_range": 30.0,
        "min_chainage": -100.0,
        "multiple_inter": "auto",
        "prc_multiple": 0.1,
        "sand_color": "dark",
        "cloud_mask_issue": True,
        "sat_list": ["L9", "S2"],
        "image_size_filter": False,
    }
    assert expected_settings == settings_dashboard.get_settings()
