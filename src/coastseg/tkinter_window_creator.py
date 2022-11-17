from tkinter import Tk


class Tkinter_Window_Creator:
    """A context manager that creates a root window for tkinter and destroys it when exited.
    The window created is withdrawn so the user does not see it and placed above all other windows."""

    def __init__(self):
        self.root = Tk()
        self.root.withdraw()  # Hide the main window.
        self.root.call(
            "wm", "attributes", ".", "-topmost", True
        )  # Raise the self.root to the top of all windows.

    def __enter__(self):
        return self.root

    def __exit__(self, type, value, traceback):
        self.root.destroy()
