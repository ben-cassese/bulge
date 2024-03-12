import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

@jax.jit
def flux_integrand(alpha, u1, u2, c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, **kwargs):
    return (
        (
            c_x3 * (-(jnp.sin(alpha) * c_y1) + jnp.cos(alpha) * c_y2)
            - c_x2 * (c_y1 + jnp.cos(alpha) * c_y3)
            + c_x1 * (c_y2 + jnp.sin(alpha) * c_y3)
        )
        * (
            3
            * u2
            * (
                (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
                + (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
            )
            ** 2
            - 4
            * (u1 + 2 * u2)
            * (
                -1
                + jnp.sqrt(
                    1
                    - (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3)
                    ** 2
                    - (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3)
                    ** 2
                )
            )
            + 2
            * (
                (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
                + (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
            )
            * (
                3
                - 3 * u1
                - 6 * u2
                + 2
                * u1
                * jnp.sqrt(
                    1
                    - (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3)
                    ** 2
                    - (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3)
                    ** 2
                )
                + 4
                * u2
                * jnp.sqrt(
                    1
                    - (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3)
                    ** 2
                    - (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3)
                    ** 2
                )
            )
        )
    ) / (
        12.0
        * (
            (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
            + (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
        )
    )
