from abc import ABC, abstractmethod


class Feature(ABC):
    @abstractmethod
    def style_layer(self):
        """
        Style the layer of the feature.
        """
        pass
