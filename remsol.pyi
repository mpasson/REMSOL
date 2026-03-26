"""Module for calculating fo electromagnetic modes in multilayer structures."""

from enum import Enum

class BoundaryCondition(Enum):
    """Boundary condition applied at the edge of a MultiLayer structure."""

    SemiInfinite = 0
    """Default. The outermost layer is treated as a semi-infinite cladding;
    the field decays evanescently away from the structure."""

    PEC = 1
    """Perfect Electric Conductor wall. The tangential electric field is forced
    to zero at this boundary."""

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

class PEC:
    """A Perfect Electric Conductor boundary marker.

    Place an instance of this class as the **first** or **last** element of the
    layer list passed to ``MultiLayer`` to impose a PEC boundary condition on
    that side of the structure.  It replaces the semi-infinite cladding on that
    side.

    Examples:
        PEC on the left::

            ml = MultiLayer([PEC(), Layer(2.0, 0.6), Layer(1.0, 1.0)])

        PEC on the right::

            ml = MultiLayer([Layer(1.0, 1.0), Layer(2.0, 0.6), PEC()])
    """

    def __init__(self) -> None: ...

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

    def __init__(self, layers: list[Layer | PEC]) -> None:
        """Create a new multilayer structure from a list of layers.

        A ``PEC`` instance may appear as the **first** or **last** element of
        ``layers`` to impose a Perfect Electric Conductor boundary condition on
        that side of the structure.  It cannot appear in any other position, and
        both ends cannot be PEC simultaneously.

        Args:
            layers: A list of ``Layer`` objects representing the layers in the
                structure, optionally preceded or followed by a single ``PEC``
                marker.
        """

    def set_left_boundary(self, bc: BoundaryCondition) -> None:
        """Set the boundary condition on the left side of the structure.

        This is an alternative to placing ``PEC()`` as the first element of the
        layer list.  Calling this method after construction achieves the same
        effect.

        Args:
            bc: ``BoundaryCondition.SemiInfinite`` (default) or
                ``BoundaryCondition.PEC``.
        """

    def set_right_boundary(self, bc: BoundaryCondition) -> None:
        """Set the boundary condition on the right side of the structure.

        This is an alternative to placing ``PEC()`` as the last element of the
        layer list.

        Args:
            bc: ``BoundaryCondition.SemiInfinite`` (default) or
                ``BoundaryCondition.PEC``.
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
