from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import os

from coastseg import common, file_utilities


DEFAULT_CLASS_MAPPING: Dict[int, str] = {
    0: "water",
    1: "whitewater",
    2: "sediment",
    3: "other",
}


@dataclass
class ModelInfo:
    """
    Stores and manages metadata for a segmentation model, including the class mapping and
    identification of water-related classes.

    This class resolves the model's directory from a name or a provided path, loads a class
    mapping from a model card file (typically in JSON format), and identifies indices
    corresponding to water-related classes (e.g., "water", "whitewater").

    Default class mapping is provided if no mapping is found in the modelcard file.
    The default mapping includes:
    - 0: "water"
    - 1: "whitewater"
    - 2: "sediment"
    - 3: "other"
    The `water_class_indices` are automatically computed based on the class names "water" and "whitewater".

    -----------------------
    Supported Initialization
    -----------------------

    1. From a model directory:
        Provide `model_directory`, optionally with `model_name`.
        Call `load()` to load the class mapping from a `modelcard.json` file in the directory.

        ⚠ The `model_directory` must contain a file matching the pattern 'modelcard.json'.
        ⚠ If no class mapping is found in the modelcard file, a default mapping is used.


        Example:
            model_info = ModelInfo(
                model_directory="models/my_model",
                model_name="my_model"
            )
            model_info.load()
            print(model_info.class_mapping)

    2. From a model name only:
        Provide `model_name`. The directory will be resolved using
        `common.get_downloaded_models_dir() / model_name`.

        Example:
            model_info = ModelInfo(model_name="AK_segformer")
            model_info.load()

    3. From a direct class mapping (no file access):
        Provide a `class_mapping` dictionary directly.
        Call `load()` to compute water-related indices based on this mapping.

        Example:
            model_info = ModelInfo(class_mapping={0: "water", 1: "land"})
            model_info.load()

    4. From a custom extractor function:
        Pass a function to `load(extractor=...)` that extracts a class mapping from a
        modelcard JSON dictionary.

        Example:
            def my_extractor(data):
                return {int(k): v for k, v in data["custom_labels"].items()}

            model_info = ModelInfo(model_directory="models/custom")
            model_info.load(extractor=my_extractor)

    Attributes:
        model_name (Optional[str]): Name of the model, used to locate the model directory if a path is not explicitly provided.
        model_directory (Optional[str]): Absolute path to the model directory. If not provided, it will be resolved using model_name.
        class_mapping (Dict[int, str]): Mapping from class indices to class labels. Loaded from a model card or provided directly.
        water_class_indices (List[int]): List of class indices corresponding to water-related classes.

    Methods:
        load(extractor: Optional[Callable[[dict], Dict[int, str]]] = None):
            Loads the class mapping and identifies water class indices. An optional extractor can be used to customize parsing of the model card.
    """

    model_name: Optional[str] = None
    model_directory: Optional[str] = None
    class_mapping: Dict[int, str] = field(default_factory=dict)
    water_class_indices: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.model_directory = self._resolve_directory()
        self.load()

    def load(self, extractor: Optional[Callable[[dict], Dict[int, str]]] = None):
        self.class_mapping = self._load_class_mapping(extractor)
        self.water_class_indices = self._find_water_classes()

    def _resolve_directory(self) -> Optional[str]:
        if self.model_directory:
            path = self.model_directory
        elif self.model_name:
            path = os.path.join(common.get_downloaded_models_dir(), self.model_name)
        else:
            return None

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        return path

    def _load_class_mapping(
        self, extractor: Optional[Callable[[dict], Dict[int, str]]] = None
    ) -> Dict[int, str]:
        if self.class_mapping:
            return self.class_mapping

        if not self.model_directory:
            return DEFAULT_CLASS_MAPPING.copy()

        # Find the model card JSON file to read the class mapping
        path = file_utilities.find_file_by_regex(
            self.model_directory, r".*modelcard\.json$"
        )
        model_card = file_utilities.read_json_file(path, raise_error=True)
        try:
            if extractor:
                return extractor(model_card)
            return self._default_extractor(model_card)
        except Exception:
            return DEFAULT_CLASS_MAPPING.copy()

    def _find_water_classes(
        self, water_names: List[str] = ["water", "whitewater"]
    ) -> List[int]:
        return [i for i, name in self.class_mapping.items() if name in water_names]

    @staticmethod
    def _default_extractor(data: dict) -> Dict[int, str]:
        for section in ("DATASET", "DATASET1"):
            if section in data and "CLASSES" in data[section]:
                return {int(k): v for k, v in data[section]["CLASSES"].items()}
        if "CLASSES" in data:
            return {int(k): v for k, v in data["CLASSES"].items()}
        return DEFAULT_CLASS_MAPPING.copy()
