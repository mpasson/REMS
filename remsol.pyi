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

    def complex_field(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        mode: int = 0,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
    ) -> FieldData:
        """Calculate the field profile of a complex mode found by the complex-plane solver.

        Mirrors the signature of :meth:`field` but uses the complex-plane solver
        (:meth:`complex_neff`) to find the effective index, then reconstructs the
        field for that mode.

        For semi-infinite boundaries the outgoing wave in the rightmost layer is
        **not** zeroed: for a complex neff the radiation condition is already encoded
        in the imaginary part, and zeroing the outgoing amplitude would give a
        physically wrong result.

        If ``Normalization.Power`` is set and the found ``neff`` has a nonzero
        imaginary part, a warning is emitted via Python's ``logging`` module and
        ``MaxField`` normalization is used instead.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the mode (TE or TM).
            mode: Zero-based mode index (same ordering as :meth:`complex_neff`).
            re_range: Optional ``(re_min, re_max)`` forwarded to the complex-plane
                solver.  Defaults to ``(Re(n_min), Re(n_max))`` across all layers.
            im_range: Optional ``(im_min, im_max)`` forwarded to the complex-plane
                solver.  Defaults to ``(-w, +w)`` where
                ``w = max(max_j |Im(n_j)|, 0.05)``.

        Returns:
            A ``FieldData`` object with all six field components on the standard
            plotting grid, normalized according to the current normalization setting
            (default: ``MaxField``).  If the requested mode does not exist, a
            ``FieldData`` with all components set to zero is returned.

        Examples:
            Fundamental TE mode of a lossy slab::

                ml = MultiLayer([
                    Layer(1.0, 1.0),
                    Layer(2.0 - 0.01j, 0.6),
                    Layer(1.0, 1.0),
                ])
                field = ml.complex_field(omega, Polarization.TE, mode=0)
        """

    def qnm_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        mode: int = 0,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Find a single quasi-normal mode (QNM) effective index.

        QNMs satisfy **outgoing-wave** boundary conditions in both semi-infinite
        cladding layers: rather than decaying evanescently away from the guiding
        region, the field radiates outward.  Their effective indices are complex
        with ``Im(neff) < 0`` (energy leaks out, so the mode decays in time).

        The solver uses the same argument-principle winding-number / Muller-polisher
        pipeline as :meth:`complex_neff`, but evaluates the S-matrix built with
        ``kz_outgoing`` (the outgoing Riemann sheet) on the first and last cladding
        layers.  Consequently the search rectangle should lie **entirely in the lower
        half-plane** (``im_max ≤ 0``).

        Default search ranges (used when the corresponding argument is ``None``):

        * **re_range**: ``(Re(n_min) + ε, Re(n_max) − ε)`` across all layers.
        * **im_range**: ``(-0.5, -1e-3)`` — entirely below the real axis.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the mode (TE or TM).
            mode: Zero-based index into the list returned by :meth:`all_qnm_neff`,
                sorted by descending ``Re(neff)``.
            re_range: Optional ``(re_min, re_max)`` for the real part of ``neff``.
                Widen to include strongly leaky modes, e.g. ``(0.0, n_max)``.
            im_range: Optional ``(im_min, im_max)`` for the imaginary part of
                ``neff``.  Must satisfy ``im_max ≤ 0``.  Widen ``im_min`` to
                capture modes with large radiation loss, e.g. ``(-2.0, -1e-3)``.

        Returns:
            ``(Re(neff), Im(neff))`` as a tuple with ``Im(neff) < 0``, or ``None``
            if the requested mode index is out of range.

        Examples:
            Leaky mode of an asymmetric slab (core over a higher-index substrate)::

                layers = [
                    Layer(1.0, 1.0),   # air cladding
                    Layer(2.0, 0.6),   # waveguide core
                    Layer(1.0, 0.5),   # thin air gap
                    Layer(2.2, 1.0),   # substrate (semi-infinite)
                ]
                ml = MultiLayer(layers)
                re, im = ml.qnm_neff(omega)
                # im < 0  →  radiation loss into the substrate

            Wider search for strongly leaky modes::

                re, im = ml.qnm_neff(omega, im_range=(-2.0, -1e-3))
        """

    def all_qnm_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
    ) -> list[tuple[float, float]]:
        """Return all quasi-normal mode (QNM) effective indices in the search rectangle.

        Uses the same argument-principle / Muller-polisher pipeline as
        :meth:`all_complex_neff`, but with outgoing-wave boundary conditions on
        the semi-infinite cladding layers.  All returned modes satisfy
        ``Im(neff) < 0``.

        Default search ranges are the same as :meth:`qnm_neff`.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the modes (TE or TM).
            re_range: Optional ``(re_min, re_max)`` for ``Re(neff)``.
            im_range: Optional ``(im_min, im_max)`` for ``Im(neff)``;
                must satisfy ``im_max ≤ 0``.

        Returns:
            A list of ``(Re(neff), Im(neff))`` tuples sorted by descending
            ``Re(neff)``.  All tuples satisfy ``Im(neff) < 0``.

        Examples:
            All QNMs of a leaky slab::

                modes = ml.all_qnm_neff(omega, Polarization.TE)
                for re, im in modes:
                    print(f"Re(neff)={re:.4f}  Im(neff)={im:.4e}")
        """

    def leaky_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        mode: int = 0,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Find a single one-sided leaky mode effective index.

        Uses **evanescent** (physical) boundary conditions on the left cladding
        and **outgoing-wave** boundary conditions on the right cladding.  This is
        the correct solver for structures where the mode is evanescently confined
        on the low-index left side and radiates into a higher-index substrate on
        the right.

        Unlike :meth:`qnm_neff`, which applies outgoing-wave conditions on
        **both** claddings (full quasi-normal mode), this solver keeps the left
        cladding in the standard evanescent regime.  The resulting mode poles:

        * Have ``Im(neff) > 0`` — with real ω and complex k∥ = neff·k₀,
          ``Im(neff) > 0`` means the field decays as it propagates along the
          waveguide (spatial decay, +x direction).
        * Have ``Re(neff)`` close to the guided-mode value of the isolated core,
          and **increasing** as the gap between core and substrate shrinks (more
          substrate overlap → higher effective index).
        * Have ``Im(neff)`` growing exponentially as the gap shrinks (stronger
          tunnelling through the gap → faster spatial decay).

        **Sign convention note:** :meth:`qnm_neff` uses complex ω at fixed real
        k∥ (temporal decay → ``Im(neff) < 0``).  This solver uses real ω with
        complex k∥ (spatial decay → ``Im(neff) > 0``).  The search rectangle
        must therefore lie entirely in the **upper** half-plane (``im_min ≥ 0``).

        Default search ranges (used when the corresponding argument is ``None``):

        * **re_range**: ``(Re(n_min) + ε, Re(n_max) − ε)`` across all layers.
        * **im_range**: ``(1e-3, 0.15)`` — entirely above the real axis.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the mode (TE or TM).
            mode: Zero-based index into the list returned by
                :meth:`all_leaky_neff`, sorted by descending ``Re(neff)``.
            re_range: Optional ``(re_min, re_max)`` for the real part of ``neff``.
            im_range: Optional ``(im_min, im_max)`` for the imaginary part of
                ``neff``.  Must satisfy ``im_min ≥ 0``.  For weakly leaky modes
                (large gap) use a shallower window such as ``(1e-8, 1e-3)``.

        Returns:
            ``(Re(neff), Im(neff))`` as a tuple with ``Im(neff) > 0``, or ``None``
            if the requested mode index is out of range.

        Examples:
            Core mode leaking into a higher-index substrate::

                layers = [
                    Layer(1.0, 1.0),   # air cladding (left)
                    Layer(2.0, 0.6),   # waveguide core
                    Layer(1.0, 0.5),   # thin air gap
                    Layer(2.2, 1.0),   # substrate (right, semi-infinite)
                ]
                ml = MultiLayer(layers)
                re, im = ml.leaky_neff(omega)
                # re ≈ isolated-core neff,  im > 0

            Weakly leaky mode (large gap, small Im(neff))::

                re, im = ml.leaky_neff(omega, im_range=(1e-8, 1e-3))
        """

    def all_leaky_neff(
        self,
        omega: float,
        polarization: Polarization = Polarization.TE,
        re_range: tuple[float, float] | None = None,
        im_range: tuple[float, float] | None = None,
    ) -> list[tuple[float, float]]:
        """Return all one-sided leaky mode effective indices in the search rectangle.

        Uses the same argument-principle / Muller-polisher pipeline as
        :meth:`all_qnm_neff`, but with evanescent BC on the left cladding and
        outgoing-wave BC on the right cladding.  All returned modes satisfy
        ``Im(neff) > 0`` (spatial decay along the propagation direction at real ω).

        Default search ranges are the same as :meth:`leaky_neff`.

        Args:
            omega: The angular frequency (real).
            polarization: The polarization of the modes (TE or TM).
            re_range: Optional ``(re_min, re_max)`` for ``Re(neff)``.
            im_range: Optional ``(im_min, im_max)`` for ``Im(neff)``;
                must satisfy ``im_min ≥ 0``.  Use a smaller lower bound such as
                ``(1e-8, 1e-3)`` to capture weakly leaky modes (large gap).

        Returns:
            A list of ``(Re(neff), Im(neff))`` tuples sorted by descending
            ``Re(neff)``.  All tuples satisfy ``Im(neff) > 0``.

        Examples:
            All leaky modes of a core-over-substrate slab::

                modes = ml.all_leaky_neff(omega, Polarization.TE)
                for re, im in modes:
                    print(f"Re(neff)={re:.6f}  Im(neff)={im:.4e}")
        """

    def index(self) -> IndexData:
        """Return the refractive index profile of the multilayer structure.

        Returns:
            An ``IndexData`` object containing the x coordinates and the real
            part of the refractive index sampled at the current ``plot_step``.
        """
```

Now let's build and check for errors:
