"""Module for calculating fo electromagnetic modes in multilayer structures."""

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

class FieldData:
    """A class representing the field data for a mode in a multilayer structure."""

    x: list[float]
    """The x values of the field data."""

    Ex: list[complex]
    """The electric field values in the x direction."""

    Ey: list[complex]
    """The electric field values in the y direction."""

    Ez: list[complex]
    """The electric field values in the z direction."""

    Hx: list[complex]
    """The magnetic field values in the x direction."""

    Hy: list[complex]
    """The magnetic field values in the y direction."""

    Hz: list[complex]
    """The magnetic field values in the z direction."""

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

    def field(
        self, omega: float, polarization: Polarization = Polarization.TE, mode: int = 0
    ) -> FieldData:
        """Calculate the field data for a given mode and polarization.

        Args:
            omega: The angular frequency of the light.
            polarization: The polarization of the light (TE or TM).
            mode: The mode number of the light.

        Returns:
            A FieldData object containing the field data for the given parameters.
        """
