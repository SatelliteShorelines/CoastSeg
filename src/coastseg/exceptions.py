class Object_Not_Found(Exception):
    """BBox_Not_Found: raised when bounding box does not exist
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, feature: str, message=""):
        self.msg = f"No {feature.lower()} found on the map.\n{message}"
        self.feature = feature
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


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
        self, msg="No shorelines were found in this region. Draw a new bounding box."
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
