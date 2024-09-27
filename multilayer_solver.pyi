from enum import Enum

class Polarization(Enum):
    """An enumeration of the two possible polarizations."""

    TE = 0
    """The transverse electric polarization."""

    TM = 1
    """The transverse magnetic polarization."""

class Layer:
    """A class representing a single layer in a multilayer structure."""

    def __init__(self, n: float, d: float) -> None:
        """Create a new layer with a given refractive index and thickness.

        Args:
            n: The refractive index of the layer.
            d: The thickness of the layer.
        """

class MultiLayer:
    """A class representing a multilayer structure."""

    def __init__(self, layers: list[Layer]) -> None:
        """Create a new multilayer structure from a list of layers.

        Args:
            layers: A list of Layer objects representing the layers in the structure.
        """

    def neff(
        self, omega: float, polarization: Polarization = Polarization.TE, mode: int = 0
    ) -> float:
        """Calculate the effective index of refraction for a given mode and polarization.

        Args:
            omega: The angular frequency of the light.
            polarization: The polarization of the light (TE or TM).
            mode: The mode number of the light.

        Returns:
            The effective index of refraction for the given parameters.
        """
