import json
import os
from typing import List
import logging

logger = logging.getLogger(__name__)


class Session:
    """
    A class representing a session, which contains sets of classes, years, and ROI IDs.
    """

    def __init__(self, name: str = None, path: str = None):
        """
        Initializes a new Session object.

        Args:
            name (str): The name of the session. Default is None.
            path (str): The path to the directory where the session will be saved. Default is None.
        """
        self.name = name
        self.path = path
        self.classes = set()
        self.years = set()
        self.roi_ids = set()
        self.roi_info = {}

    def get_session_data(self) -> dict:
        session_data = {
            "name": self.name,
            "path": self.path,
            "classes": list(self.classes),
            "years": list(self.years),
            "roi_ids": list(self.roi_ids),
        }
        return session_data

    def get_roi_info(self, roi_id: str = None):
        if roi_id:
            return self.roi_info.get(roi_id, "")
        return self.roi_info

    def set_roi_info(self, new_roi_info: dict):
        return self.roi_info.update(new_roi_info)

    def add_classes(self, class_names: List[str]):
        """
        Adds one or more class names to the session.

        Args:
            class_names (str or iterable): The name(s) of the class(es) to add.
        """
        if isinstance(class_names, str):
            self.classes.add(class_names)
        else:
            self.classes.update(class_names)

    def add_roi_ids(self, roi_ids: List[str]):
        """
        Adds one or more ROI IDs to the session.

        Args:
            roi_ids (str or iterable): The ROI ID(s) to add.
        """
        if isinstance(roi_ids, str):
            self.roi_ids.add(roi_ids)
        else:
            self.roi_ids.update(roi_ids)

    def find_session_file(self, path: str, filename: str = "session.json"):
        # if session.json is found in main directory then session path was identified
        session_path = os.path.join(path, filename)
        if os.path.isfile(session_path):
            return session_path
        else:
            parent_directory = os.path.dirname(path)
            json_path = os.path.join(parent_directory, filename)
            if os.path.isfile(json_path):
                return json_path
            else:
                raise ValueError(
                    f"File '{filename}' not found in the parent directory: {parent_directory} or path"
                )

    def load(self, path: str):
        """
        Loads a session from a directory.

        Args:
            path (str): The path to the session directory.
        """
        json_path = self.find_session_file(path, "session.json")
        with open(json_path, "r") as f:
            session_data = json.load(f)
            self.name = session_data.get("name")
            self.path = session_data.get("path")
            self.classes = set(session_data.get("classes", []))
            self.roi_ids = set(session_data.get("roi_ids", []))

    def save(self, path):
        """
        Saves the session to a directory.

        Args:
            path (str): The path to the directory where the session will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        session_data = {
            "name": self.name,
            "path": path,
            "classes": list(self.classes),
            "roi_ids": list(self.roi_ids),
        }

        with open(os.path.join(path, "session.json"), "w") as f:
            json.dump(session_data, f, indent=4)

    def __str__(self):
        """
        Returns a string representation of the session.

        Returns:
            str: A string representation of the session.
        """
        return f"Session: {self.name}\nPath: {self.path}\nClasses: {self.classes}\nROI IDs: {self.roi_ids}\n"
