import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def rho_xx(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_xx - p_xz**2 / (4.0 * p_zz)


def rho_xy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_xy - (p_xz * p_yz) / (2.0 * p_zz)


def rho_x0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_x0 - (p_xz * p_z0) / (2.0 * p_zz)


def rho_yy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_yy - p_yz**2 / (4.0 * p_zz)


def rho_y0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_y0 - (p_yz * p_z0) / (2.0 * p_zz)


def rho_00(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_00 - p_z0**2 / (4.0 * p_zz)


@jax.jit
def planet_2d_coeffs(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, **kwargs
):
    return {
        "rho_xx": rho_xx(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_xy": rho_xy(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_x0": rho_x0(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_yy": rho_yy(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_y0": rho_y0(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_00": rho_00(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
    }
