import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 - rho_x0 + rho_xx

def _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2*rho_xy + 2*rho_y0

def _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2 + 2*rho_00 - 2*rho_xx + 4*rho_yy

def _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return 2*rho_xy + 2*rho_y0

def _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 + rho_x0 + rho_xx

@jax.jit
def single_intersection_points(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
    t4 = _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t3 = _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t2 = _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t1 = _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t0 = _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)

    polys = jnp.array([t4, t3, t2, t1, t0])
    roots = jnp.roots(polys, strip_zeros=False) # strip_zeros must be False to jit

    ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
    xs = jnp.where(ts != 999, (1-ts**2)/(1+ts**2), ts)
    ys = jnp.where(ts != 999, 2*ts/(1+ts**2), ts)
    return xs, ys

@jax.jit
def multiple_intersection_points(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
    t4 = _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t3 = _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t2 = _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t1 = _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t0 = _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)

    polys = jnp.array([t4, t3, t2, t1, t0]).T

    roots = jax.vmap(lambda x: jnp.roots(x, strip_zeros=False))(polys)

    ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
    xs = jnp.where(ts != 999, (1-ts**2)/(1+ts**2), ts)
    ys = jnp.where(ts != 999, 2*ts/(1+ts**2), ts)
    return xs, ys