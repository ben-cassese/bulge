import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def x(a, e, f, Omega, i, omega):
    return (
        a
        * (-1 + e**2)
        * (
            jnp.sin(f)
            * (
                jnp.cos(Omega) * jnp.sin(omega)
                + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
            )
            + jnp.cos(f)
            * (
                -(jnp.cos(omega) * jnp.cos(Omega))
                + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )
    ) / (1 + e * jnp.cos(f))


def y(a, e, f, Omega, i, omega):
    return -(
        (
            a
            * (-1 + e**2)
            * (
                jnp.cos(i) * jnp.cos(Omega) * jnp.sin(f + omega)
                + jnp.cos(f + omega) * jnp.sin(Omega)
            )
        )
        / (1 + e * jnp.cos(f))
    )


def z(a, e, f, Omega, i, omega):
    return -(
        (a * (-1 + e**2) * jnp.sin(i) * jnp.sin(f + omega)) / (1 + e * jnp.cos(f))
    )


def x_bound_1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return (
        -(rho_xy * rho_y0)
        + 2 * rho_x0 * rho_yy
        + 2
        * jnp.sqrt(rho_yy)
        * jnp.sqrt(
            -(rho_x0 * rho_xy * rho_y0)
            + rho_xx * rho_y0**2
            + rho_x0**2 * rho_yy
            + rho_00 * (rho_xy**2 - 4 * rho_xx * rho_yy)
        )
    ) / (rho_xy**2 - 4 * rho_xx * rho_yy)


def x_bound_2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -(
        (
            rho_xy * rho_y0
            - 2 * rho_x0 * rho_yy
            + 2
            * jnp.sqrt(rho_yy)
            * jnp.sqrt(
                rho_00 * rho_xy**2
                - rho_x0 * rho_xy * rho_y0
                + rho_xx * rho_y0**2
                + rho_x0**2 * rho_yy
                - 4 * rho_00 * rho_xx * rho_yy
            )
        )
        / (rho_xy**2 - 4 * rho_xx * rho_yy)
    )


def y_bound_1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -(
        (
            rho_x0 * rho_xy
            - 2 * rho_xx * rho_y0
            + 2
            * jnp.sqrt(rho_xx)
            * jnp.sqrt(
                rho_00 * rho_xy**2
                - rho_x0 * rho_xy * rho_y0
                + rho_xx * rho_y0**2
                + rho_x0**2 * rho_yy
                - 4 * rho_00 * rho_xx * rho_yy
            )
        )
        / (rho_xy**2 - 4 * rho_xx * rho_yy)
    )


def y_bound_2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return (
        -(rho_x0 * rho_xy)
        + 2 * rho_xx * rho_y0
        + 2
        * jnp.sqrt(rho_xx)
        * jnp.sqrt(
            -(rho_x0 * rho_xy * rho_y0)
            + rho_xx * rho_y0**2
            + rho_x0**2 * rho_yy
            + rho_00 * (rho_xy**2 - 4 * rho_xx * rho_yy)
        )
    ) / (rho_xy**2 - 4 * rho_xx * rho_yy)


@jax.jit
def x_bounds(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
    return jnp.array(
        [
            x_bound_1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00),
            x_bound_2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00),
        ]
    )


@jax.jit
def y_bounds(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
    return jnp.array(
        [
            y_bound_1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00),
            y_bound_2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00),
        ]
    )

@jax.jit
def skypos(a, e, f, Omega, i, omega, **kwargs):
    return jnp.array(
        [
            x(a, e, f, Omega, i, omega),
            y(a, e, f, Omega, i, omega),
            z(a, e, f, Omega, i, omega),
        ]
    )
