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

    * **Complex-plane solver** (``complex_neff`` / ``all_complex_neff``): uses
      the argument principle (winding-number method) to locate zeros of the
      scattering-matrix determinant in a rectangle of the complex *neff* plane,
      then polishes each zero with Muller's method.  Finds lossy guided modes
      (when layers have ``Im(n) != 0``) and leaky modes (when ``Re(neff)`` falls
      below the cladding index).
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

        Uses the fast real-axis scan.  Only finds modes with purely real
        ``neff``; for lossy or leaky structures use :meth:`complex_neff`.

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
    ) -> tuple[float, float] | None:
        """Find a single complex effective index using the 2-D complex-plane solver.

        The solver locates zeros of the scattering-matrix determinant inside the
        rectangle ``re_range × im_range`` of the complex *neff* plane using the
        argument principle, then polishes each zero with Muller's method.

        Default search ranges (used when the corresponding argument is ``None``):

        * **re_range**: ``(Re(n_min), Re(n_max))`` across all layers — the same
          bounds used by the real-axis solver.  This automatically spans both
          guided (``Re(neff) > Re(n_clad)``) and leaky regimes.
        * **im_range**: ``(-w, +w)`` where
          ``w = max(max_j |Im(n_j)|, 1e-3)``.  For lossless structures this
          gives a small but nonzero imaginary window so that weakly leaky modes
          are not missed.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the mode (TE or TM).
            mode: Zero-based index into the list returned by
                :meth:`all_complex_neff`, sorted by descending ``Re(neff)``.
            re_range: Optional ``(re_min, re_max)`` for the real part of
                ``neff``.  Pass a wider range, e.g. ``(0.0, n_max)``, to
                include strongly leaky modes.
            im_range: Optional ``(im_min, im_max)`` for the imaginary part of
                ``neff``.  For a lossy structure with no leaky modes of interest
                you can restrict this to e.g. ``(-0.1, 0.0)``.

        Returns:
            ``(Re(neff), Im(neff))`` as a tuple, or ``None`` if the requested
            mode index is out of range.

        Examples:
            Fundamental TE mode of a lossless slab (should match ``neff``)::

                ml = MultiLayer([Layer(1.0, 1.0), Layer(2.0, 0.6), Layer(1.0, 1.0)])
                re, im = ml.complex_neff(omega, Polarization.TE, mode=0)

            Lossy core::

                ml = MultiLayer([
                    Layer(1.0, 1.0),
                    Layer(2.0 + 0.01j, 0.6),
                    Layer(1.0, 1.0),
                ])
                re, im = ml.complex_neff(omega)

            Leaky mode search with a wider real range::

                re, im = ml.complex_neff(omega, re_range=(0.5, 2.0), im_range=(-0.5, 0.0))
        """

    def all_complex_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
    ) -> list[tuple[float, float]]:
        """Return all complex effective indices found in the search rectangle.

        Uses the same argument-principle / Muller solver as :meth:`complex_neff`.

        Default search ranges are the same as :meth:`complex_neff`.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the modes (TE or TM).
            re_range: Optional ``(re_min, re_max)`` for ``Re(neff)``.
            im_range: Optional ``(im_min, im_max)`` for ``Im(neff)``.

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

    def field_complex(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        neff_re: float = 0.0,
        neff_im: float = 0.0,
    ) -> FieldData:
        """Calculate the field profile for a mode with a given complex effective index.

        Unlike :meth:`field`, which uses the real-axis solver to find neff internally,
        this method accepts an explicit complex neff (as returned by
        :meth:`complex_neff` or :meth:`all_complex_neff`) and reconstructs the field
        for that mode.

        For semi-infinite boundaries the outgoing wave in the rightmost layer is
        **not** zeroed: for a complex neff the radiation condition is already encoded
        in the imaginary part, and zeroing the outgoing amplitude would give a
        physically wrong result.

        If ``Normalization.Power`` is set and ``neff_im != 0``, a warning is emitted
        via Python's ``logging`` module and ``MaxField`` normalization is used instead.

        Args:
            omega: The angular frequency (real, same units as used for ``neff``).
            polarization: The polarization of the mode (TE or TM).
            neff_re: Real part of the complex effective index.
            neff_im: Imaginary part of the complex effective index.

        Returns:
            A ``FieldData`` object with all six field components on the standard
            plotting grid, normalized according to the current normalization setting
            (default: ``MaxField``).

        Examples:
            Field for a lossy guided mode::

                neff_re, neff_im = ml.complex_neff(omega, Polarization.TE)
                field = ml.field_complex(omega, Polarization.TE, neff_re, neff_im)
        """

    def index(self) -> IndexData:
        """Return the refractive index profile of the multilayer structure.

        Returns:
            An ``IndexData`` object containing the x coordinates and the real
            part of the refractive index sampled at the current ``plot_step``.
        """
```

Now let's build and check for errors:
