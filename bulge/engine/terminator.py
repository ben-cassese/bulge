import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bulge.engine.skypos import skypos


def _t_00(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        p_00
        + (
            (p_x0 * x_c + p_y0 * y_c + p_z0 * z_c)
            * (
                -(p_xz * p_z0 * x_c)
                + p_x0 * p_zz * x_c
                - p_yz * p_z0 * y_c
                + p_y0 * p_zz * y_c
                - p_z0 * p_zz * z_c
            )
        )
        / (p_xz * x_c + p_yz * y_c + 2 * p_zz * z_c) ** 2
    )


def _t_x0(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return -(
        (
            y_c
            * (
                p_xz * (p_xz * p_y0 - p_x0 * p_yz + p_xy * p_z0) * x_c
                - 2 * p_x0 * p_xy * p_zz * x_c
                + p_yz * (p_xz * p_y0 - p_x0 * p_yz + p_xy * p_z0) * y_c
                - 2 * p_xy * p_y0 * p_zz * y_c
            )
            + 2
            * p_xx
            * x_c
            * (
                p_xz * p_z0 * x_c
                + p_yz * p_z0 * y_c
                - 2 * p_zz * (p_x0 * x_c + p_y0 * y_c)
            )
            + 2
            * (p_xz * p_z0 - 2 * p_x0 * p_zz)
            * (p_xz * x_c + p_yz * y_c)
            * z_c
            + 2 * p_zz * (p_xz * p_z0 - 2 * p_x0 * p_zz) * z_c**2
        )
        / (p_xz * x_c + p_yz * y_c + 2 * p_zz * z_c) ** 2
    )


def _t_xx(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        4 * p_xx**2 * p_zz * x_c**2
        + (p_xy * y_c + p_xz * z_c)
        * (
            -(p_xz**2 * x_c)
            + p_xy * p_zz * y_c
            - p_xz * (p_yz * y_c + p_zz * z_c)
        )
        + p_xx
        * (
            -(p_xz**2 * x_c**2)
            + 4 * p_xy * p_zz * x_c * y_c
            + 4 * p_xz * p_zz * x_c * z_c
            + (p_yz * y_c + 2 * p_zz * z_c) ** 2
        )
    ) / (p_xz * x_c + p_yz * y_c + 2 * p_zz * z_c) ** 2


def _t_xy(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        -2
        * (
            p_xx
            * x_c
            * (
                p_xz * p_yz * x_c
                - 2 * p_xy * p_zz * x_c
                + (p_yz**2 - 4 * p_yy * p_zz) * y_c
            )
            + y_c
            * (
                p_xz**2 * p_yy * x_c
                + p_xz * p_yy * p_yz * y_c
                - p_xy * p_zz * (p_xy * x_c + 2 * p_yy * y_c)
            )
            + (p_xz * p_yz - 2 * p_xy * p_zz) * (p_xz * x_c + p_yz * y_c) * z_c
            + p_zz * (p_xz * p_yz - 2 * p_xy * p_zz) * z_c**2
        )
    ) / (p_xz * x_c + p_yz * y_c + 2 * p_zz * z_c) ** 2


def _t_yy(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        p_xz**2 * p_yy * x_c**2
        + p_xy**2 * p_zz * x_c**2
        - p_xy * (p_yz**2 - 4 * p_yy * p_zz) * x_c * y_c
        - p_xz * x_c * (p_xy * p_yz * x_c + (p_yz**2 - 4 * p_yy * p_zz) * z_c)
        - (p_yz**2 - 4 * p_yy * p_zz)
        * (p_yy * y_c**2 + p_yz * y_c * z_c + p_zz * z_c**2)
    ) / (p_xz * x_c + p_yz * y_c + 2 * p_zz * z_c) ** 2


def _t_y0(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        p_xz**2 * p_y0 * x_c**2
        + p_x0
        * x_c
        * (2 * p_xy * p_zz * x_c - p_yz**2 * y_c + 4 * p_yy * p_zz * y_c)
        - p_xz
        * x_c
        * (
            p_x0 * p_yz * x_c
            + p_xy * p_z0 * x_c
            - p_y0 * p_yz * y_c
            + 2 * p_yy * p_z0 * y_c
            + 2 * p_yz * p_z0 * z_c
            - 4 * p_y0 * p_zz * z_c
        )
        - (p_yz * p_z0 - 2 * p_y0 * p_zz)
        * (
            p_xy * x_c * y_c
            + 2 * p_yy * y_c**2
            + 2 * z_c * (p_yz * y_c + p_zz * z_c)
        )
    ) / (p_xz * x_c + p_yz * y_c + 2 * p_zz * z_c) ** 2


@jax.jit
def terminator_2d_coeffs(
    p_xx,
    p_xy,
    p_xz,
    p_x0,
    p_yy,
    p_yz,
    p_y0,
    p_zz,
    p_z0,
    p_00,
    x_c,
    y_c,
    z_c,
    **kwargs,
):
    return {
        "t_xx": _t_xx(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "t_xy": _t_xy(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "t_x0": _t_x0(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "t_yy": _t_yy(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "t_y0": _t_y0(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "t_00": _t_00(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
    }


def _t4(
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    t_xx,
    t_xy,
    t_x0,
    t_yy,
    t_y0,
    t_00,
):
    return rho_xx * (
        rho_yy**2 * t_xx**2
        - rho_xy * rho_yy * t_xx * t_xy
        + t_yy * (rho_xy**2 * t_xx - rho_xx * rho_xy * t_xy + rho_xx**2 * t_yy)
        + rho_xx * rho_yy * (t_xy**2 - 2 * t_xx * t_yy)
    )


def _t3(
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    t_xx,
    t_xy,
    t_x0,
    t_yy,
    t_y0,
    t_00,
):
    return rho_xx * (
        2 * rho_y0 * rho_yy * t_xx**2
        + 2 * rho_xx * rho_yy * t_x0 * t_xy
        - rho_x0 * rho_yy * t_xx * t_xy
        + rho_xx * rho_y0 * t_xy**2
        + rho_xy**2 * t_xx * t_y0
        - 2 * rho_xx * rho_yy * t_xx * t_y0
        + rho_xx
        * (-2 * rho_y0 * t_xx - rho_x0 * t_xy + 2 * rho_xx * t_y0)
        * t_yy
        - rho_xy
        * (
            rho_yy * t_x0 * t_xx
            + rho_y0 * t_xx * t_xy
            + rho_xx * t_xy * t_y0
            + rho_xx * t_x0 * t_yy
            - 2 * rho_x0 * t_xx * t_yy
        )
    )


def _t2(
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    t_xx,
    t_xy,
    t_x0,
    t_yy,
    t_y0,
    t_00,
):
    return rho_xx * (
        t_xx
        * (
            rho_xy**2 * (-1 + t_00)
            + (rho_y0**2 + 2 * (-1 + rho_00) * rho_yy) * t_xx
            - rho_x0 * (rho_yy * t_x0 + rho_y0 * t_xy)
            + rho_xy
            * (-(rho_y0 * t_x0) + t_xy - rho_00 * t_xy + 2 * rho_x0 * t_y0)
            + rho_x0**2 * t_yy
        )
        + rho_xx**2 * (t_y0**2 + 2 * (-1 + t_00) * t_yy)
        + rho_xx
        * (
            rho_yy * (t_x0**2 - 2 * (-1 + t_00) * t_xx)
            + 2 * rho_y0 * t_x0 * t_xy
            - t_xy**2
            + rho_00 * t_xy**2
            - 2 * rho_y0 * t_xx * t_y0
            - rho_x0 * t_xy * t_y0
            + rho_xy * (t_xy - t_00 * t_xy - t_x0 * t_y0)
            - rho_x0 * t_x0 * t_yy
            + 2 * t_xx * t_yy
            - 2 * rho_00 * t_xx * t_yy
        )
    )


def _t1(
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    t_xx,
    t_xy,
    t_x0,
    t_yy,
    t_y0,
    t_00,
):
    return rho_xx * (
        2 * rho_xx**2 * (-1 + t_00) * t_y0
        + t_xx
        * (
            -((-1 + rho_00) * (rho_xy * t_x0 - 2 * rho_y0 * t_xx))
            + rho_x0
            * (2 * rho_xy * (-1 + t_00) - rho_y0 * t_x0 + t_xy - rho_00 * t_xy)
            + rho_x0**2 * t_y0
        )
        + rho_xx
        * (
            rho_xy * (t_x0 - t_00 * t_x0)
            + rho_y0 * (t_x0**2 + 2 * t_xx - 2 * t_00 * t_xx)
            + rho_x0 * t_xy
            - rho_x0 * t_00 * t_xy
            - 2 * t_x0 * t_xy
            + 2 * rho_00 * t_x0 * t_xy
            - rho_x0 * t_x0 * t_y0
            + 2 * t_xx * t_y0
            - 2 * rho_00 * t_xx * t_y0
        )
    )


def _t0(
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    t_xx,
    t_xy,
    t_x0,
    t_yy,
    t_y0,
    t_00,
):
    return rho_xx * (
        rho_xx**2 * (-1 + t_00) ** 2
        - rho_x0 * rho_xx * (-1 + t_00) * t_x0
        + t_xx
        * (
            rho_x0 * (rho_x0 * (-1 + t_00) + t_x0 - rho_00 * t_x0)
            + (-1 + rho_00) ** 2 * t_xx
        )
        + (-1 + rho_00) * rho_xx * (t_x0**2 - 2 * (-1 + t_00) * t_xx)
    )

@jax.jit
def terminator_quartic_coeffs(
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    t_xx,
    t_xy,
    t_x0,
    t_yy,
    t_y0,
    t_00,
):
    return {
        "t4": _t4(
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            t_xx,
            t_xy,
            t_x0,
            t_yy,
            t_y0,
            t_00,
        ),
        "t3": _t3(
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            t_xx,
            t_xy,
            t_x0,
            t_yy,
            t_y0,
            t_00,
        ),
        "t2": _t2(
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            t_xx,
            t_xy,
            t_x0,
            t_yy,
            t_y0,
            t_00,
        ),
        "t1": _t1(
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            t_xx,
            t_xy,
            t_x0,
            t_yy,
            t_y0,
            t_00,
        ),
        "t0": _t0(
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            t_xx,
            t_xy,
            t_x0,
            t_yy,
            t_y0,
            t_00,
        ),
    }


def terminator_intersection_xs(
    y,
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    t_xx,
    t_xy,
    t_x0,
    t_yy,
    t_y0,
    t_00,
):
    return (
        -(t_xx * (-1 + rho_00 + y * (rho_y0 + rho_yy * y)))
        + rho_xx * (-1 + t_00 + y * (t_y0 + t_yy * y))
    ) / (t_xx * (rho_x0 + rho_xy * y) - rho_xx * (t_x0 + t_xy * y))
