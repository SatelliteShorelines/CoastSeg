class Object_Not_Found(Exception):
    """Object_Not_Found: raised when bounding box does not exist
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, feature: str, message=""):
        self.msg = f"No {feature.lower()} found on the map.\n{message}"
        self.feature = feature
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class WarningException(Exception):
    """WarningException: class for all exceptions that contain custom instructions
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, message: str = "", instructions: str = ""):
        self.msg = f"{message}"
        self.instructions = f"{instructions}"
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"
    
class WarningMissingDirsException(Exception):
    """WarningMissingDirsException: class for all exceptions that are raised when something is wrong with the configuration, but the program can still continue
    Args:
        Exception: Inherits from the WarningException class
    """

    def __init__(self, message: str = "", instructions: str = "",missing_dirs: dict = {},styled_instructions:str=""):
        self.msg = f"{message}"
        self.instructions = f"{instructions}"
        self.missing_dirs = missing_dirs
        self.styled_instructions = styled_instructions
        super().__init__(self.msg)
    
    # set the missing folders for each roi id
    def get_styled_message(self):
        # format the missing directories into a string
        missing_dirs_str = "</br>".join([f"<span style='color: red'><i><b>{roi_id}</b> was missing the folder '<u>{folder}</u>'</i></span>" for roi_id, folder in self.missing_dirs.items()])
        message=f"{self.msg}</br> \n{missing_dirs_str}"
        return message
    
    def get_instructions(self):
        if self.styled_instructions:
            return self.styled_instructions
        return self.instructions

    def __str__(self):
        # format the missing directories into a string
        if len(self.missing_dirs) == 0:
            return f"{self.msg}"
        missing_dirs_str = "\n".join([f"- {roi_id} was missing the folder '{folder}'" for roi_id, folder in self.missing_dirs.items()])
        message=f"\n{self.msg}\n\n{missing_dirs_str}\n\n{self.instructions}"
        return message


class No_Extracted_Shoreline(Exception):
    """No_Extracted_Shoreline: raised when ROI id does not have a shoreline to extract
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(
        self, id: int = None, msg=f"The ROI does not have a shoreline to extract."
    ):
        self.msg = msg
        if id is not None:
            self.msg = f"The ROI id {id} does not have a shoreline to extract."
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class Id_Not_Found(Exception):
    """Id_Not_Found: raised when ROI id does not exist
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, id: int = None, msg="The ROI id does not exist."):
        self.msg = msg
        if id is not None:
            self.msg = f"The ROI id {id} does not exist."
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class Duplicate_ID_Exception(Exception):
    """Id_Not_Found: raised when duplicate IDs are detected in a feature
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(
        self,
        feature_type: str = "feature",
        msg="Duplicate ids were detected.Do you want to override these IDs?",
    ):
        self.msg = msg
        if id is not None:
            self.msg = f"Duplicate ids for {feature_type} were detected.Do you want to override these IDs?"
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class BBox_Not_Found(Exception):
    """BBox_Not_Found: raised when bounding box does not exist
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(
        self, msg="The bounding box does not exist. Draw a bounding box first"
    ):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class Shoreline_Not_Found(Exception):
    """Shoreline_Not_Found: raised when shoreline is not found in bounding box
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(
        self,
        msg="CoastSeg currently does not have shorelines available in this region. Try drawing a new bounding box somewhere else.",
    ):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class BboxTooLargeError(Exception):
    """BboxTooLargeError: raised when bounding box is larger than MAX_BBOX_SIZE
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, msg="The bounding box was too large."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class BboxTooSmallError(Exception):
    """BboxTooLargeError: raised when bounding box is smaller than MIN_BBOX_SIZE
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, msg="The bounding box was too small."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class InvalidSize(Exception):
    """
    Raised when an object is outside the allowed size range.

    Attributes:
        feature_name (str): The name of the feature or object that caused the exception.
        msg (str): A descriptive error message.
        provided_size (int, optional): The size that caused the exception.
        min_size (int, optional): The minimum allowed size.
        max_size (int, optional): The maximum allowed size.
        error_code (int, optional): An error code to categorize the exception.
    """

    def __init__(
        self,
        msg: str,
        feature_name: str,
        provided_size: int = None,
        min_size: int = None,
        max_size: int = None,
        error_code: int = None,
    ):
        self.feature_name = feature_name
        self.provided_size = provided_size
        self.min_size = min_size
        self.max_size = max_size
        self.error_code = error_code
        self.msg = f"{feature_name} : {msg}"

        if self.provided_size is not None:
            self.msg += f" | Provided Size: {self.provided_size}"

        if self.min_size is not None and self.max_size is not None:
            self.msg += (
                f" | Allowed Size Range: [{self.min_size}, {self.max_size}] meters² "
            )

        if self.error_code is not None:
            self.msg += f" | Error Code: {self.error_code}"

        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class InvalidGeometryType(Exception):
    """
    Raised when a feature's geometry type doesn't match the expected type(s).

    Attributes:
        feature_name (str): The name of the feature causing the exception.
        msg (str): A descriptive error message.
        expected_geom_types (set): A set of expected geometry types.
        wrong_geom_type (str): The geometry type that caused the exception.
    """

    def __init__(
        self,
        msg: str,
        feature_name: str,
        expected_geom_types: set,
        wrong_geom_type: str,
        help_msg: str = None,
        error_code: int = None,
    ):
        self.feature_name = feature_name
        self.expected_geom_types = expected_geom_types
        self.wrong_geom_type = wrong_geom_type
        self.help_msg = help_msg
        self.error_code = error_code
        self.msg = f"{feature_name} : {msg}. Expected: {', '.join(expected_geom_types)}, but got: {wrong_geom_type}."

        if self.help_msg is not None:
            self.msg += f" \n {self.help_msg}"

        if self.error_code is not None:
            self.msg += f" | Error Code: {self.error_code}"

        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class DownloadError(Exception):
    """DownloadError: raised when a download error occurs.
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, file):
        msg = f"\n ERROR\nShoreline file:'{file}' is not online.\nPlease raise an issue on GitHub with the shoreline name.\n https://github.com/SatelliteShorelines/CoastSeg/issues"
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"
