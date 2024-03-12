import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 - rho_x0 + rho_xx

def t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2*rho_xy + 2*rho_y0

def t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2 + 2*rho_00 - 2*rho_xx + 4*rho_yy

def t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return 2*rho_xy + 2*rho_y0

def t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 + rho_x0 + rho_xx

@jax.jit
def intersection_points(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
    t4_coeff = t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t3_coeff = t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t2_coeff = t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t1_coeff = t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t0_coeff = t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)

    polys = jnp.array([t4_coeff, t3_coeff, t2_coeff, t1_coeff, t0_coeff])
    roots = jnp.roots(polys, strip_zeros=False) # strip_zeros must be False to jit

    ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
    xs = jnp.where(ts != 999, (1-ts**2)/(1+ts**2), ts)
    ys = jnp.where(ts != 999, 2*ts/(1+ts**2), ts)
    return xs, ys