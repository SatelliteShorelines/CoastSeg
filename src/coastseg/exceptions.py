from typing import Optional


class Object_Not_Found(Exception):
    """
    Raised when a required object is not found on the map.

    Args:
        feature: Name of the missing feature.
        message: Additional error message.
    """

    def __init__(self, feature: str, message: str = ""):
        self.msg = f"No {feature.lower()} found on the map.\n{message}"
        self.feature = feature
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class WarningException(Exception):
    """
    Base class for exceptions with custom instructions.

    Args:
        message: Error message.
        instructions: Custom instructions.
    """

    def __init__(self, message: str = "", instructions: str = ""):
        self.msg = f"{message}"
        self.instructions = f"{instructions}"
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class WarningMissingDirsException(WarningException):
    """
    Raised for configuration issues that allow program continuation.

    Args:
        message: Error message.
        instructions: Instructions for user.
        missing_dirs: Dict of missing directories.
        styled_instructions: Styled instructions.
    """

    def __init__(
        self,
        message: str = "",
        instructions: str = "",
        missing_dirs: dict = {},
        styled_instructions: str = "",
    ):
        self.msg = f"{message}"
        self.instructions = f"{instructions}"
        self.missing_dirs = missing_dirs
        self.styled_instructions = styled_instructions
        super().__init__(self.msg)

    # set the missing folders for each roi id
    def get_styled_message(self):
        # format the missing directories into a string
        missing_dirs_str = "</br>".join(
            [
                f"<span style='color: red'><i><b>{roi_id}</b> was missing the folder '<u>{folder}</u>'</i></span>"
                for roi_id, folder in self.missing_dirs.items()
            ]
        )
        message = f"{self.msg}</br> \n{missing_dirs_str}"
        return message

    def get_instructions(self):
        if self.styled_instructions:
            return self.styled_instructions
        return self.instructions

    def __str__(self):
        # format the missing directories into a string
        if len(self.missing_dirs) == 0:
            return f"{self.msg}"
        missing_dirs_str = "\n".join(
            [
                f"- {roi_id} was missing the folder '{folder}'"
                for roi_id, folder in self.missing_dirs.items()
            ]
        )
        message = f"\n{self.msg}\n\n{missing_dirs_str}\n\n{self.instructions}"
        return message


class No_Extracted_Shoreline(Exception):
    """
    Raised when ROI has no shoreline to extract.

    Args:
        id: ROI ID or empty string.
        msg: Error message.
    """

    def __init__(
        self,
        id: Optional[str] = "",
        msg: str = "The ROI does not have a shoreline to extract.",
    ):
        self.msg = msg
        if id:
            self.msg = f"The ROI id {id} does not have a shoreline to extract."
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class Id_Not_Found(Exception):
    """
    Raised when ROI ID does not exist.

    Args:
        id: ROI ID or None.
        msg: Error message.
    """

    def __init__(
        self, id: Optional[int] = None, msg: str = "The ROI id does not exist."
    ):
        self.msg = msg
        if id is not None:
            self.msg = f"The ROI id {id} does not exist."
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class BboxTooLargeError(Exception):
    """
    Raised when bounding box exceeds maximum size.

    Args:
        msg: Error message.
    """

    def __init__(self, msg: str = "The bounding box was too large."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class BboxTooSmallError(Exception):
    """
    Raised when bounding box is smaller than minimum size.

    Args:
        msg: Error message.
    """

    def __init__(self, msg: str = "The bounding box was too small."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class InvalidSize(Exception):
    """
    Raised when object size is outside allowed range.

    Args:
        msg: Error message.
        feature_name: Name of the feature.
        provided_size: Actual size provided.
        min_size: Minimum allowed size.
        max_size: Maximum allowed size.
        error_code: Optional error code.
    """

    def __init__(
        self,
        msg: str,
        feature_name: str,
        provided_size: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        error_code: Optional[int] = None,
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
    Raised when geometry type doesn't match expected types.

    Args:
        msg: Error message.
        feature_name: Name of the feature.
        expected_geom_types: Set of expected geometry types.
        wrong_geom_type: Actual geometry type.
        help_msg: Optional help message.
        error_code: Optional error code.
    """

    def __init__(
        self,
        msg: str,
        feature_name: str,
        expected_geom_types: set,
        wrong_geom_type: str,
        help_msg: Optional[str] = None,
        error_code: Optional[int] = None,
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
    """
    Raised when a download error occurs.

    Args:
        file: Name of the file that failed to download.
    """

    def __init__(self, file: str):
        msg = f"\n ERROR\nShoreline file:'{file}' is not online.\nPlease raise an issue on GitHub with the shoreline name.\n https://github.com/SatelliteShorelines/CoastSeg/issues"
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"
