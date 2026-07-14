import functools
from typing import Literal, Union
import warnings

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import FloatLike, ScalarLike

from linpde_gp import domains, functions
from linpde_gp.linfuncops import diffops
from linpde_gp.typing import DomainLike
from linpde_gp.utils import to_real2, from_real2

from ._bvp import DirichletBoundaryCondition, BoundaryValueProblem
from ._linear_pde import LinearPDE


def _is_function_G(value) -> bool:
    """Return True if a G_real/G_imag argument is function-valued."""
    return isinstance(value, pn.functions.Function)


def _coerce_G_to_field(
    G: Union[FloatLike, pn.functions.Function],
    domain_shape: tuple,
) -> pn.functions.Function:
    """Promote a (possibly scalar) G_real/G_imag value to a Function over the
    spatial domain. Used internally to build the k²(x) field uniformly when
    at least one of G_real / G_imag is function-valued."""
    if _is_function_G(G):
        if G.input_shape != domain_shape:
            raise ValueError(
                f"G.input_shape ({G.input_shape}) must match domain_shape "
                f"({domain_shape})."
            )
        if G.output_shape != ():
            raise ValueError(
                f"G must be scalar-valued, got output_shape={G.output_shape}."
            )
        return G
    return functions.Constant(domain_shape, value=float(G))


def _ensure_function_real2(fn: pn.functions.Function) -> pn.functions.Function:
    if fn.output_shape and fn.output_shape[-1] == 2:
        return fn

    def _stack_real_imag(x):
        values = fn(x)
        return to_real2(values)

    return functions.JaxLambdaFunction(
        _stack_real_imag,
        input_shape=fn.input_shape,
        output_shape=fn.output_shape + (2,),
        vectorize=False,
    )


def _ensure_array_real2(values: Union[float, complex, np.ndarray]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape and arr.shape[-1] == 2:
        return arr
    return to_real2(arr)


class HelmholtzEquation(LinearPDE):
    """(In)homogeneous Helmholtz equation.
    
    The equation takes the form:
    (G' + iG'') * ∇²u(f) = -ρω²u(f)
    
    Which can be rewritten as:
    ∇²u + k²u = 0
    
    where k² = ρω²/(G' + iG'') is the wave number squared.
    """
    
    def __init__(
        self,
        domain: DomainLike,
        rho: FloatLike,
        omega: FloatLike,
        G_real: Union[FloatLike, pn.functions.Function],
        G_imag: Union[FloatLike, pn.functions.Function] = 0.0,
        rhs: pn.functions.Function | None = None,
        *,
        complex_representation: Literal["none", "real2"] = "none",
    ):
        """Initialize the Helmholtz equation.

        Parameters
        ----------
        domain : DomainLike
            The spatial domain
        rho : FloatLike
            Density parameter ρ
        omega : FloatLike
            Angular frequency ω
        G_real : FloatLike | pn.functions.Function
            Real part of the modulus G'. Either a scalar (homogeneous medium)
            or a ``pn.functions.Function`` representing G'(x) on the spatial
            domain (heterogeneous medium).
        G_imag : FloatLike | pn.functions.Function
            Imaginary part of the modulus G''. Same scalar/function semantics
            as G_real.
        rhs : pn.functions.Function | None
            Right-hand side function (default is zero for homogeneous equation)
        complex_representation : {"none", "real2"}
            Optional real-valued representation of the complex Helmholtz
            operator.

        Notes
        -----
        When both ``G_real`` and ``G_imag`` are scalars the operator is built
        with a constant :math:`k^2 = \\rho\\omega^2 / (G' + iG'')`, exactly as
        in earlier versions.

        When either ``G_real`` or ``G_imag`` is a
        ``pn.functions.Function``, the Helmholtz operator is built with a
        spatially-varying :math:`k^2(x)` field via
        :meth:`HelmholtzOperator.from_coefficient_field` (or its real2
        variant). In that case ``self.k_squared`` is a ``Function``, not a
        scalar.
        """
        self._rho = float(rho)
        self._omega = float(omega)
        self._G_real = G_real if _is_function_G(G_real) else float(G_real)
        self._G_imag = G_imag if _is_function_G(G_imag) else float(G_imag)
        if complex_representation not in ("none", "real2"):
            raise ValueError("`complex_representation` must be 'none' or 'real2'.")
        self._complex_representation = complex_representation

        is_variable_G = _is_function_G(G_real) or _is_function_G(G_imag)
        self._is_variable_G = is_variable_G

        domain = domains.asdomain(domain)

        if not is_variable_G:
            # --- Scalar code path (unchanged) -----------------------------
            G_complex = complex(self._G_real, self._G_imag)
            k_squared_complex = (self._rho * self._omega**2) / G_complex
            if abs(k_squared_complex.imag) < 1e-12:
                self._k_squared = k_squared_complex.real
            else:
                self._k_squared = k_squared_complex

            if self._complex_representation == "real2":
                helmholtz_op = diffops.HelmholtzReal2Operator(
                    domain_shape=domain.shape,
                    k_squared=self._k_squared,
                )
            else:
                helmholtz_op = diffops.HelmholtzOperator(
                    domain_shape=domain.shape,
                    k_squared=self._k_squared,
                )
        else:
            # --- Variable-coefficient code path ---------------------------
            G_real_field = _coerce_G_to_field(G_real, domain.shape)
            G_imag_field = _coerce_G_to_field(G_imag, domain.shape)

            rho_omega2 = self._rho * (self._omega ** 2)

            # Probe to decide whether the resulting k²(x) is real or complex
            try:
                sample_x = np.zeros(domain.shape, dtype=np.double)
                g_re = float(np.asarray(G_real_field(sample_x)))
                g_im = float(np.asarray(G_imag_field(sample_x)))
                k2_probe = rho_omega2 / complex(g_re, g_im)
                # k²(x) is real iff G_imag is identically zero. We use the
                # probe value only as a hint; the field itself produces the
                # correct dtype at evaluation time.
                _is_pure_real_probe = abs(k2_probe.imag) < 1e-12 and not _is_function_G(
                    G_imag
                )
            except Exception:  # pylint: disable=broad-except
                _is_pure_real_probe = False

            if _is_pure_real_probe:
                # Real-valued k²(x) field: just rho_omega2 / G_real(x)
                def _k2(x, _Gr=G_real_field, _rho_w2=rho_omega2):
                    return _rho_w2 / _Gr.jax(x) if hasattr(_Gr, "jax") else (
                        _rho_w2 / jnp.asarray(np.asarray(_Gr(np.asarray(x))))
                    )

                k_squared_field = functions.JaxLambdaFunction(
                    _k2,
                    input_shape=domain.shape,
                    output_shape=(),
                    vectorize=True,
                )
            else:
                # Complex k²(x) field: rho_omega2 / (Gr(x) + i Gi(x))
                def _k2(
                    x,
                    _Gr=G_real_field,
                    _Gi=G_imag_field,
                    _rho_w2=rho_omega2,
                ):
                    gr = _Gr.jax(x) if hasattr(_Gr, "jax") else jnp.asarray(
                        np.asarray(_Gr(np.asarray(x)))
                    )
                    gi = _Gi.jax(x) if hasattr(_Gi, "jax") else jnp.asarray(
                        np.asarray(_Gi(np.asarray(x)))
                    )
                    return _rho_w2 / (gr + 1j * gi)

                k_squared_field = functions.JaxLambdaFunction(
                    _k2,
                    input_shape=domain.shape,
                    output_shape=(),
                    vectorize=True,
                )

            self._k_squared = k_squared_field

            if self._complex_representation == "real2":
                helmholtz_op = diffops.HelmholtzReal2Operator.from_coefficient_field(
                    domain_shape=domain.shape,
                    k_squared_field=k_squared_field,
                )
            else:
                helmholtz_op = diffops.HelmholtzOperator.from_coefficient_field(
                    domain_shape=domain.shape,
                    k_squared_field=k_squared_field,
                )
        
        # Homogeneous equation, rhs should be zero with correct output shape
        if rhs is None:
            rhs = functions.Zero(domain.shape, output_shape=helmholtz_op.output_codomain_shape)
        
        super().__init__(
            domain=domain,
            diffop=helmholtz_op,
            rhs=rhs,
        )
    
    @property
    def k_squared(self) -> Union[float, complex]:
        """Wave number squared."""
        return self._k_squared
    
    @property
    def rho(self) -> float:
        """Density parameter."""
        return self._rho
    
    @property
    def omega(self) -> float:
        """Angular frequency."""
        return self._omega
    
    @property
    def G_complex(self):
        """Complex modulus G' + iG''.

        Returns ``complex(G_real, G_imag)`` for scalar inputs. When either
        ``G_real`` or ``G_imag`` is a Function this property returns a
        ``pn.functions.Function`` that evaluates to ``G_real(x) + 1j*G_imag(x)``
        pointwise.
        """
        if not self._is_variable_G:
            return complex(self._G_real, self._G_imag)

        G_real_field = _coerce_G_to_field(self._G_real, self.domain.shape)
        G_imag_field = _coerce_G_to_field(self._G_imag, self.domain.shape)

        def _g_complex(
            x, _Gr=G_real_field, _Gi=G_imag_field
        ):
            gr = _Gr.jax(x) if hasattr(_Gr, "jax") else jnp.asarray(
                np.asarray(_Gr(np.asarray(x)))
            )
            gi = _Gi.jax(x) if hasattr(_Gi, "jax") else jnp.asarray(
                np.asarray(_Gi(np.asarray(x)))
            )
            return gr + 1j * gi

        return functions.JaxLambdaFunction(
            _g_complex,
            input_shape=self.domain.shape,
            output_shape=(),
            vectorize=True,
        )

    @property
    def is_variable_coefficient(self) -> bool:
        """True if either G_real or G_imag was passed as a Function."""
        return self._is_variable_G

    @property
    def complex_representation(self) -> Literal["none", "real2"]:
        return self._complex_representation


class HelmholtzEquationDirichletProblem(BoundaryValueProblem):
    """Helmholtz equation with Dirichlet boundary conditions."""
    
    def __init__(
        self,
        domain: DomainLike,
        rho: FloatLike,
        omega: FloatLike,
        G_real: Union[FloatLike, pn.functions.Function],
        G_imag: Union[FloatLike, pn.functions.Function] = 0.0,
        boundary_values: pn.functions.Function | tuple | list | None = None,
        rhs: pn.functions.Function | None = None,
        solution: pn.functions.Function | None = None,
        *,
        complex_representation: Literal["none", "real2"] = "none",
    ):
        """Initialize the Helmholtz Dirichlet problem.
        
        Parameters
        ----------
        domain : DomainLike
            The spatial domain
        rho : FloatLike
            Density parameter ρ
        omega : FloatLike
            Angular frequency ω
        G_real : FloatLike
            Real part of the modulus G'
        G_imag : FloatLike
            Imaginary part of the modulus G''
        boundary_values : pn.functions.Function | tuple | list | None
            Values on the boundary. Can be:
            - None: zero boundary conditions
            - pn.functions.Function: same function applied to all boundaries
            - tuple/list of length 2: (left_value, right_value) for 1D domains
        rhs : pn.functions.Function | None
            Right-hand side function (default is zero)
        solution : pn.functions.Function | None
            Known analytical solution if available
        """
        is_real2_mode = complex_representation == "real2"

        if is_real2_mode and rhs is not None:
            rhs = _ensure_function_real2(rhs)

        pde = HelmholtzEquation(
            domain=domain,
            rho=rho,
            omega=omega,
            G_real=G_real,
            G_imag=G_imag,
            rhs=rhs,
            complex_representation=complex_representation,
        )

        is_real2 = pde.complex_representation == "real2"
        
        # Set boundary conditions
        if boundary_values is None:
            boundary_values = functions.Zero(
                pde.domain.shape, pde.diffop.input_codomain_shape
            )
        elif is_real2:
            if isinstance(boundary_values, pn.functions.Function):
                boundary_values = _ensure_function_real2(boundary_values)
            elif isinstance(boundary_values, (tuple, list)):
                converted = []
                for value in boundary_values:
                    if isinstance(value, pn.functions.Function):
                        converted.append(_ensure_function_real2(value))
                    else:
                        converted.append(_ensure_array_real2(value))
                boundary_values = tuple(converted)
            else:
                boundary_values = _ensure_array_real2(boundary_values)
        
        if isinstance(boundary_values, (tuple, list)) and len(boundary_values) == 2:
            # For 1D domains with 2 boundary values, interpret as (left, right)
            if len(pde.domain.boundary) != 2:
                raise ValueError(
                    f"Tuple boundary values only supported for 1D domains with 2 boundaries, "
                    f"but got {len(pde.domain.boundary)} boundaries"
                )
            
            left_value, right_value = boundary_values
            
            # Create separate boundary conditions for each boundary point
            boundary_conditions = []
            for i, boundary_part in enumerate(pde.domain.boundary):
                # First boundary (leftmost) gets left_value, second gets right_value
                bc_value = left_value if i == 0 else right_value

                bc = DirichletBoundaryCondition(boundary_part, bc_value)
                boundary_conditions.append(bc)
            
            boundary_conditions = tuple(boundary_conditions)
            
            # Store the original boundary values for reference
            self._boundary_values = boundary_values
        
        elif isinstance(boundary_values, (tuple, list)):
            if len(boundary_values) != len(pde.domain.boundary):
                raise ValueError(
                    f"Expected {len(pde.domain.boundary)} boundary values, "
                    f"got {len(boundary_values)}."
                )
            boundary_conditions = tuple(
                DirichletBoundaryCondition(boundary_part, value)
                for boundary_part, value in zip(pde.domain.boundary, boundary_values)
            )
            
        else:
            boundary_conditions = tuple(
                DirichletBoundaryCondition(boundary_part, boundary_values)
                for boundary_part in pde.domain.boundary
            )
            
            # Store the boundary values
            self._boundary_values = boundary_values
        
        # For 1D interval domains, try to find analytical solutions
        if solution is None and isinstance(domain, domains.Interval):
            if pde.is_variable_coefficient:
                # The closed-form complex-exponential branch assumes constant
                # k². For a variable k²(x) field there is no general closed
                # form, so we deliberately leave `solution = None` and warn
                # the caller.
                warnings.warn(
                    "Analytical solution is not available for variable-"
                    "coefficient Helmholtz BVPs (G_real and/or G_imag are "
                    "function-valued). `solution` will remain None.",
                    UserWarning,
                    stacklevel=2,
                )
            # Check if we can construct an analytical solution
            elif isinstance(boundary_values, functions.Zero):
                # Trivial solution for homogeneous equation with zero BC
                solution = functions.Zero(domain.shape, output_shape=())
            else:
                # Try to construct analytical solution for non-zero boundary conditions
                try:
                    # Extract boundary values for analytical solution
                    if hasattr(self, '_boundary_values') and isinstance(self._boundary_values, (tuple, list)):
                        # Use the stored tuple values directly
                        bc_values = tuple(self._boundary_values)
                    elif isinstance(boundary_values, pn.functions.Function):
                        # Evaluate function at boundaries
                        a, b = domain
                        bc_values = (boundary_values(a), boundary_values(b))
                    elif isinstance(boundary_values, (int, float, np.number)):
                        # Single scalar - use for both boundaries
                        bc_values = (boundary_values, boundary_values)
                    else:
                        # Default fallback
                        bc_values = (0.0, 0.0)

                    # Convert stacked real2 boundary values back to complex scalars
                    if is_real2:
                        converted_bc_values = []
                        for value in bc_values:
                            arr = np.asarray(value)
                            if arr.shape and arr.shape[-1] == 2:
                                converted_bc_values.append(from_real2(arr))
                            else:
                                converted_bc_values.append(value)
                        bc_values = tuple(converted_bc_values)
                    
                    # Only create analytical solution for homogeneous equation (rhs = 0)
                    if rhs is None or isinstance(rhs, functions.Zero):
                        solution = Solution_HelmholtzEquation_DirichletProblem_1D_ComplexExponential(
                            domain=domain,
                            k_squared=pde.k_squared,
                            boundary_values=bc_values,
                        )
                except ValueError as e:
                    # Singular system or other issue - no analytical solution available
                    pass

        if is_real2 and solution is not None:
            solution = _ensure_function_real2(solution)
        
        super().__init__(
            pde=pde,
            boundary_conditions=boundary_conditions,
            solution=solution,
        )


# Example analytical solution for 1D Helmholtz with specific boundary conditions
class Solution_HelmholtzEquation_DirichletProblem_1D_ComplexExponential(
    functions.JaxFunction
):
    """Analytical solution for 1D Helmholtz equation with complex exponential form.
    
    For the equation ∇²u + k²u = 0 on [a, b], the general solution is:
    u(x) = A*exp(ikx) + B*exp(-ikx)
    
    where A and B are determined by boundary conditions.
    """
    
    def __init__(
        self,
        domain: domains.Interval,
        k_squared: ScalarLike,
        boundary_values: tuple[ScalarLike, ScalarLike],
    ):
        self._domain = domain
        self._k_squared = k_squared
        self._k = np.sqrt(np.complex128(k_squared))
        self._u_a, self._u_b = boundary_values
        
        a, b = domain
        
        # Solve for coefficients A and B from boundary conditions
        # u(a) = A*exp(ika) + B*exp(-ika) = u_a
        # u(b) = A*exp(ikb) + B*exp(-ikb) = u_b
        
        exp_ika = np.exp(1j * self._k * a)
        exp_neg_ika = np.exp(-1j * self._k * a)
        exp_ikb = np.exp(1j * self._k * b)
        exp_neg_ikb = np.exp(-1j * self._k * b)
        
        # Matrix form: [exp(ika), exp(-ika)] [A] = [u_a]
        #              [exp(ikb), exp(-ikb)] [B]   [u_b]
        
        det = exp_ika * exp_neg_ikb - exp_ikb * exp_neg_ika
        
        if np.abs(det) < 1e-10:
            raise ValueError("Singular system - boundary conditions may be incompatible")
        
        self._A = (self._u_a * exp_neg_ikb - self._u_b * exp_neg_ika) / det
        self._B = (self._u_b * exp_ika - self._u_a * exp_ikb) / det
        
        # Store domain bounds for JAX computation
        self._a = a
        self._b = b
        
        super().__init__(input_shape=(), output_shape=())
    
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return (
            self._A * np.exp(1j * self._k * x) + 
            self._B * np.exp(-1j * self._k * x)
        )
    
    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        k = jnp.sqrt(jnp.asarray(self._k_squared, dtype=jnp.complex128))
        A = jnp.asarray(self._A, dtype=jnp.complex128)
        B = jnp.asarray(self._B, dtype=jnp.complex128)
        x = jnp.asarray(x, dtype=jnp.complex128)
        return A * jnp.exp(1j * k * x) + B * jnp.exp(-1j * k * x)
