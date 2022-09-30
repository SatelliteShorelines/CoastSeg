class BboxTooLargeError(Exception):
    """BboxTooLargeError: raised when bounding box is larger than MAX_BBOX_SIZE
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, msg="The bounding box was too large."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return (f"{self.msg}")


class BboxTooSmallError(Exception):
    """BboxTooLargeError: raised when bounding box is smaller than MIN_BBOX_SIZE
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, msg="The bounding box was too small."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return (f"{self.msg}")
    
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
        return (f"{self.msg}")
