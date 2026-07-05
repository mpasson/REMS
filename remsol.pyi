"""Module for calculating electromagnetic modes in multilayer structures."""

from enum import Enum

class BoundaryCondition(Enum):
    """Boundary condition applied at the edge of a MultiLayer structure."""

    SemiInfinite = 0
    """Default. The outermost layer is treated as a semi-infinite cladding;
    the field decays evanescently away from the structure."""

    PEC = 1
    """Perfect Electric Conductor wall. The tangential electric field is forced
    to zero at this boundary."""

    Outgoing = 2
    """Outgoing-wave (radiating) boundary condition.  The field propagates
    outward rather than decaying evanescently.  Use on one or both claddings
    to find leaky modes or quasi-normal modes (QNMs) via
    :meth:`MultiLayer.complex_neff` / :meth:`MultiLayer.all_complex_neff`.

    * Both claddings ``Outgoing`` → QNMs (``Im(neff) < 0``, temporal decay).
    * Left ``SemiInfinite`` + right ``Outgoing`` → one-sided leaky modes
      (``Im(neff) > 0``, spatial decay along the guide).
    """

class Normalization(Enum):
    """Normalization convention used for field reconstruction."""

    MaxField = 0
    """Default. The field is normalized so that the maximum of the total electric
    field amplitude equals 1: max(sqrt(|Ex|² + |Ey|² + |Ez|²)) = 1.
    Works for all modes including leaky and lossy modes."""

    Power = 1
    """The field is normalized so that the absolute value of the integrated
    z-component of the Poynting vector equals 1. Only physically meaningful for
    lossless guided modes. If requested for a complex neff, a warning is emitted
    and MaxField normalization is used instead."""

class Polarization(Enum):
    """An enumeration of the two possible polarizations."""

    TE = 0
    """The transverse electric polarization."""

    TM = 1
    """The transverse magnetic polarization."""

class Layer:
    """A class representing a single layer in a multilayer structure.

    The refractive index may be real (lossless medium) or complex (lossy or
    gain medium).  Passing a plain Python ``float`` is fully backward-compatible
    with previous versions of the library.
    """

    def __init__(self, n: float | complex, d: float) -> None:
        """Create a new layer with a given refractive index and thickness.

        Args:
            n: The refractive index of the layer.  May be a plain ``float``
               (lossless medium, backward-compatible) or a Python ``complex``
               (lossy / gain medium, e.g. ``1.5 + 0.01j``).
            d: The thickness of the layer in the same units used for the
               vacuum wavevector ``omega`` (typically µm).
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
    """The x coordinates of the field data."""

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

class IndexData:
    """A class representing the refractive index profile of a multilayer structure."""

    x: list[float]
    """The x coordinates of the index profile."""

    n: list[float]
    """The real part of the refractive index at each x coordinate."""

class MultiLayer:
    """A class representing a multilayer structure.

    Modes are found by one of two solvers:

    * **Real-axis solver** (``neff`` / ``all_neff``): fast scan along the real
      wavevector axis.  Finds only lossless guided modes.  Use this for the
      common case of real refractive indices.

    * **Complex solver** (``complex_neff`` / ``all_complex_neff``): exact
      unsquared interface condition for a two-layer structure; argument-principle
      search of the scattering-matrix determinant for larger stacks. Finds lossy
      guided modes and leaky or quasi-normal modes.
    """

    plot_step: float
    """Step size used when sampling the field and index profiles (default: 1e-3)."""

    normalization: Normalization
    """Normalization convention used for field reconstruction (default: ``MaxField``)."""

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
        effect.  Set to ``BoundaryCondition.Outgoing`` to use outgoing-wave
        boundary conditions (for QNM / leaky-mode searches via
        :meth:`complex_neff`).

        Args:
            bc: ``BoundaryCondition.SemiInfinite`` (default),
                ``BoundaryCondition.PEC``, or ``BoundaryCondition.Outgoing``.
        """

    def set_right_boundary(self, bc: BoundaryCondition) -> None:
        """Set the boundary condition on the right side of the structure.

        This is an alternative to placing ``PEC()`` as the last element of the
        layer list.  Set to ``BoundaryCondition.Outgoing`` to use outgoing-wave
        boundary conditions (for QNM / leaky-mode searches via
        :meth:`complex_neff`).

        Args:
            bc: ``BoundaryCondition.SemiInfinite`` (default),
                ``BoundaryCondition.PEC``, or ``BoundaryCondition.Outgoing``.
        """

    def set_normalization(self, norm: Normalization) -> None:
        """Set the normalization convention used for field reconstruction.

        Args:
            norm: ``Normalization.MaxField`` (default) or ``Normalization.Power``.
        """

    def neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        mode: int = 0,
    ) -> float | None:
        """Calculate the effective index of refraction for a given guided mode.

        Uses the fast real-axis scan for multilayers. A two-layer interface uses
        its exact TM condition and is returned only when ``neff`` is real. For
        lossy or leaky structures use :meth:`complex_neff`.

        Args:
            omega: The angular frequency of the light.
            polarization: The polarization of the light (TE or TM).
            mode: The mode number (0 = fundamental, 1 = first higher-order, …).

        Returns:
            The real effective index of refraction, or ``None`` if the requested
            mode number exceeds the number of supported guided modes.
        """

    def all_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
    ) -> list[float]:
        """Return all real effective indices supported by the structure.

        Uses the fast real-axis scan.  For lossy or leaky structures use
        :meth:`all_complex_neff`.

        Args:
            omega: The angular frequency of the light.
            polarization: The polarization of the light (TE or TM).

        Returns:
            A list of effective indices sorted from highest to lowest.
        """

    def complex_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        mode: int = 0,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
        left_bc: BoundaryCondition | None = None,
        right_bc: BoundaryCondition | None = None,
    ) -> tuple[float, float] | None:
        """Find a single complex effective index using the 2-D complex-plane solver.

        For exactly two layers, the exact unsquared interface condition is used
        instead of the S-matrix determinant. Omitted ranges do not constrain the
        candidate; supplied ``re_range`` and ``im_range`` values filter it
        independently.

        For larger stacks, boundary conditions used to build the S-matrix are
        controlled by ``left_bc`` / ``right_bc``. If omitted, stored BCs are
        used (set via :meth:`set_left_boundary` / :meth:`set_right_boundary`).

        | ``left_bc``    | ``right_bc``   | Mode type                              |
        |----------------|----------------|----------------------------------------|
        | ``SemiInfinite``| ``SemiInfinite``| Guided / lossy guided (``Im ≈ 0``)    |
        | ``Outgoing``   | ``Outgoing``   | Quasi-normal modes (``Im(neff) < 0``)  |
        | ``SemiInfinite``| ``Outgoing``   | One-sided leaky (``Im(neff) > 0``)     |

        The default ``im_range`` adapts to the active boundary conditions:

        * ``SemiInfinite`` + ``SemiInfinite`` → near-real window.
        * Both ``Outgoing`` → lower half-plane ``(-0.15, -1e-3)`` for QNMs.
        * One ``Outgoing`` → upper half-plane ``(1e-3, 0.15)`` for leaky modes.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the mode (TE or TM).
            mode: Zero-based index into the list returned by
                :meth:`all_complex_neff`, sorted by descending ``Re(neff)``.
            re_range: Optional ``(re_min, re_max)`` for the real part of ``neff``.
            im_range: Optional ``(im_min, im_max)`` for the imaginary part of
                ``neff``.
            left_bc: Optional BC override for the left cladding (overrides the
                stored BC for this call only).
            right_bc: Optional BC override for the right cladding (overrides the
                stored BC for this call only).

        Returns:
            ``(Re(neff), Im(neff))`` as a tuple, or ``None`` if the requested
            mode index is out of range.

        Examples:
            Guided mode (same as ``neff``):

                re, im = ml.complex_neff(omega)

            Quasi-normal mode (QNM) with per-call BC override::

                from remsol import BoundaryCondition
                re, im = ml.complex_neff(
                    omega,
                    left_bc=BoundaryCondition.Outgoing,
                    right_bc=BoundaryCondition.Outgoing,
                )
                # im < 0

            One-sided leaky mode::

                re, im = ml.complex_neff(
                    omega,
                    left_bc=BoundaryCondition.SemiInfinite,
                    right_bc=BoundaryCondition.Outgoing,
                )
                # im > 0
        """

    def all_complex_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
        left_bc: BoundaryCondition | None = None,
        right_bc: BoundaryCondition | None = None,
    ) -> list[tuple[float, float]]:
        """Return all complex effective indices found in the search rectangle.

        Same boundary-condition logic as :meth:`complex_neff`.  See that
        method’s documentation for the ``left_bc`` / ``right_bc`` parameter
        description and the table of mode types.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the modes (TE or TM).
            re_range: Optional ``(re_min, re_max)`` for ``Re(neff)``.
            im_range: Optional ``(im_min, im_max)`` for ``Im(neff)``.
            left_bc: Optional BC override for the left cladding.
            right_bc: Optional BC override for the right cladding.

        Returns:
            A list of ``(Re(neff), Im(neff))`` tuples sorted by descending
            ``Re(neff)``.
        """

    def field(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        mode: int = 0,
    ) -> FieldData:
        """Calculate the field data for a given guided mode.

        Args:
            omega: The angular frequency of the light.
            polarization: The polarization of the light (TE or TM).
            mode: The mode number (0 = fundamental).

        Returns:
            A ``FieldData`` object containing the full vectorial field
            distribution (Ex, Ey, Ez, Hx, Hy, Hz) normalized according to the
            current normalization setting (default: ``MaxField``, i.e. the
            maximum total electric field amplitude equals 1).  If the requested
            mode number exceeds the number of supported modes, a ``FieldData``
            with all field components set to zero is returned.
        """

    def complex_field(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        mode: int = 0,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
        left_bc: BoundaryCondition | None = None,
        right_bc: BoundaryCondition | None = None,
    ) -> FieldData:
        """Calculate the field profile of a complex mode found by the complex-plane solver.

        Mirrors the signature of :meth:`field` but uses the complex-plane solver
        (:meth:`complex_neff`) to find the effective index, then reconstructs the
        field for that mode.

        The ``left_bc`` / ``right_bc`` arguments control the S-matrix boundary
        conditions used during the mode search (same semantics as
        :meth:`complex_neff`).  Field reconstruction always uses the TMM with
        ``SemiInfinite`` conditions regardless of the BCs used in the search.

        A physical two-layer TM interface uses the closed-form field relations,
        enforcing continuity of ``Hy`` and ``epsilon * Ez``. Larger stacks
        retain the TMM coefficient reconstruction.

        If ``Normalization.Power`` is set and the found ``neff`` has a nonzero
        imaginary part, a warning is emitted via Python’s ``logging`` module and
        ``MaxField`` normalization is used instead.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the mode (TE or TM).
            mode: Zero-based mode index (same ordering as :meth:`complex_neff`).
            re_range: Optional ``(re_min, re_max)`` forwarded to the complex-plane
                solver.
            im_range: Optional ``(im_min, im_max)`` forwarded to the complex-plane
                solver.
            left_bc: Optional BC override for the left cladding.
            right_bc: Optional BC override for the right cladding.

        Returns:
            A ``FieldData`` object with all six field components on the standard
            plotting grid.  If the requested mode does not exist, a ``FieldData``
            with all components set to zero is returned.
        """

    def index(self) -> IndexData:
        """Return the refractive index profile of the multilayer structure.

        Returns:
            An ``IndexData`` object containing the x coordinates and the real
            part of the refractive index sampled at the current ``plot_step``.
        """
```

Now let's build and check for errors:
