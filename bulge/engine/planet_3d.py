import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def p_xx(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        (
            jnp.cos(omega)
            * (
                jnp.cos(Omega) * jnp.sin(theta)
                + jnp.cos(i) * jnp.cos(theta) * jnp.sin(Omega)
            )
            + jnp.sin(omega)
            * (
                jnp.cos(theta) * jnp.cos(Omega)
                - jnp.cos(i) * jnp.sin(theta) * jnp.sin(Omega)
            )
        )
        ** 2
        / (-1 + f2) ** 2
        + (
            jnp.sin(i) * jnp.sin(phi) * jnp.sin(Omega)
            + jnp.cos(phi)
            * jnp.sin(theta)
            * (
                jnp.cos(Omega) * jnp.sin(omega)
                + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
            )
            + jnp.cos(theta)
            * jnp.cos(phi)
            * (
                -(jnp.cos(omega) * jnp.cos(Omega))
                + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )
        ** 2
        + (
            jnp.cos(Omega) * jnp.sin(theta) * jnp.sin(phi) * jnp.sin(omega)
            + (
                -(jnp.cos(phi) * jnp.sin(i))
                + jnp.cos(i) * jnp.cos(omega) * jnp.sin(theta) * jnp.sin(phi)
            )
            * jnp.sin(Omega)
            + jnp.cos(theta)
            * jnp.sin(phi)
            * (
                -(jnp.cos(omega) * jnp.cos(Omega))
                + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )
        ** 2
        / (-1 + f1) ** 2
    ) / r**2


def p_xy(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        (
            16
            * (
                jnp.cos(Omega) ** 2
                * jnp.sin(i)
                * jnp.sin(theta)
                * jnp.sin(2 * phi)
                * jnp.sin(omega)
                - 2
                * jnp.cos(phi) ** 2
                * jnp.cos(Omega)
                * jnp.sin(i) ** 2
                * jnp.sin(Omega)
                - 2
                * jnp.cos(phi)
                * jnp.sin(i)
                * jnp.sin(theta)
                * jnp.sin(phi)
                * jnp.sin(omega)
                * jnp.sin(Omega) ** 2
                + jnp.cos(i)
                * jnp.sin(phi)
                * (
                    jnp.cos(2 * omega)
                    * jnp.cos(2 * Omega)
                    * jnp.sin(2 * theta)
                    * jnp.sin(phi)
                    + jnp.cos(theta) ** 2
                    * jnp.cos(2 * Omega)
                    * jnp.sin(phi)
                    * jnp.sin(2 * omega)
                    - jnp.cos(2 * Omega)
                    * jnp.sin(theta) ** 2
                    * jnp.sin(phi)
                    * jnp.sin(2 * omega)
                    + 4
                    * jnp.cos(phi)
                    * jnp.cos(omega)
                    * jnp.cos(Omega)
                    * jnp.sin(i)
                    * jnp.sin(theta)
                    * jnp.sin(Omega)
                    + 4
                    * jnp.cos(theta)
                    * jnp.cos(phi)
                    * jnp.cos(Omega)
                    * jnp.sin(i)
                    * jnp.sin(omega)
                    * jnp.sin(Omega)
                )
                - jnp.cos(theta)
                * jnp.cos(omega)
                * (
                    jnp.cos(2 * Omega) * jnp.sin(i) * jnp.sin(2 * phi)
                    + 4
                    * jnp.cos(Omega)
                    * jnp.sin(theta)
                    * jnp.sin(phi) ** 2
                    * jnp.sin(omega)
                    * jnp.sin(Omega)
                )
                + jnp.cos(theta) ** 2
                * jnp.cos(omega) ** 2
                * jnp.sin(phi) ** 2
                * jnp.sin(2 * Omega)
                + jnp.sin(theta) ** 2
                * jnp.sin(phi) ** 2
                * jnp.sin(omega) ** 2
                * jnp.sin(2 * Omega)
                - jnp.cos(i) ** 2
                * jnp.sin(phi) ** 2
                * jnp.sin(theta + omega) ** 2
                * jnp.sin(2 * Omega)
            )
        )
        / (-1 + f1) ** 2
        - 32
        * (
            -(
                jnp.cos(theta) ** 2
                * jnp.cos(phi) ** 2
                * jnp.cos(omega) ** 2
                * jnp.cos(Omega)
                * jnp.sin(Omega)
            )
            + jnp.cos(Omega)
            * jnp.sin(i) ** 2
            * jnp.sin(phi) ** 2
            * jnp.sin(Omega)
            + jnp.cos(theta)
            * jnp.cos(phi)
            * jnp.cos(omega)
            * (
                -(jnp.cos(Omega) ** 2 * jnp.sin(i) * jnp.sin(phi))
                - jnp.cos(i)
                * jnp.cos(phi)
                * jnp.cos(2 * Omega)
                * jnp.sin(theta + omega)
                + jnp.sin(i) * jnp.sin(phi) * jnp.sin(Omega) ** 2
                + jnp.cos(phi)
                * jnp.sin(theta)
                * jnp.sin(omega)
                * jnp.sin(2 * Omega)
            )
            + jnp.cos(phi)
            * jnp.sin(i)
            * jnp.sin(phi)
            * (
                jnp.cos(Omega) ** 2 * jnp.sin(theta) * jnp.sin(omega)
                - jnp.sin(theta) * jnp.sin(omega) * jnp.sin(Omega) ** 2
                + jnp.cos(i) * jnp.sin(theta + omega) * jnp.sin(2 * Omega)
            )
            + jnp.cos(phi) ** 2
            * (
                jnp.cos(i)
                * jnp.cos(2 * Omega)
                * jnp.sin(theta)
                * jnp.sin(omega)
                * jnp.sin(theta + omega)
                - jnp.cos(Omega)
                * jnp.sin(theta) ** 2
                * jnp.sin(omega) ** 2
                * jnp.sin(Omega)
                + (
                    jnp.cos(i) ** 2
                    * jnp.sin(theta + omega) ** 2
                    * jnp.sin(2 * Omega)
                )
                / 2.0
            )
        )
        + (
            4 * jnp.sin(i - 2 * (theta + omega - Omega))
            - 4 * jnp.sin(i + 2 * (theta + omega - Omega))
            + 2 * jnp.sin(2 * (i - Omega))
            + jnp.sin(2 * (i - theta - omega - Omega))
            + 6 * jnp.sin(2 * (theta + omega - Omega))
            + jnp.sin(2 * (i + theta + omega - Omega))
            + 4 * jnp.sin(2 * Omega)
            - 2 * jnp.sin(2 * (i + Omega))
            - jnp.sin(2 * (i - theta - omega + Omega))
            - 6 * jnp.sin(2 * (theta + omega + Omega))
            - jnp.sin(2 * (i + theta + omega + Omega))
            + 4 * jnp.sin(i - 2 * (theta + omega + Omega))
            - 4 * jnp.sin(i + 2 * (theta + omega + Omega))
        )
        / (-1 + f2) ** 2
    ) / (16.0 * r**2)


def p_xz(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        2
        * (
            -(
                (
                    jnp.cos(theta + omega)
                    * jnp.sin(i)
                    * (
                        jnp.cos(omega)
                        * (
                            jnp.cos(Omega) * jnp.sin(theta)
                            + jnp.cos(i) * jnp.cos(theta) * jnp.sin(Omega)
                        )
                        + jnp.sin(omega)
                        * (
                            jnp.cos(theta) * jnp.cos(Omega)
                            - jnp.cos(i) * jnp.sin(theta) * jnp.sin(Omega)
                        )
                    )
                )
                / (-1 + f2) ** 2
            )
            + (
                jnp.cos(i) * jnp.sin(phi)
                - jnp.cos(phi) * jnp.sin(i) * jnp.sin(theta + omega)
            )
            * (
                jnp.sin(i) * jnp.sin(phi) * jnp.sin(Omega)
                + jnp.cos(phi)
                * jnp.sin(theta)
                * (
                    jnp.cos(Omega) * jnp.sin(omega)
                    + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
                )
                + jnp.cos(theta)
                * jnp.cos(phi)
                * (
                    -(jnp.cos(omega) * jnp.cos(Omega))
                    + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
                )
            )
            - (
                (
                    jnp.cos(i) * jnp.cos(phi)
                    + jnp.sin(i) * jnp.sin(phi) * jnp.sin(theta + omega)
                )
                * (
                    jnp.cos(Omega)
                    * jnp.sin(theta)
                    * jnp.sin(phi)
                    * jnp.sin(omega)
                    + (
                        -(jnp.cos(phi) * jnp.sin(i))
                        + jnp.cos(i)
                        * jnp.cos(omega)
                        * jnp.sin(theta)
                        * jnp.sin(phi)
                    )
                    * jnp.sin(Omega)
                    + jnp.cos(theta)
                    * jnp.sin(phi)
                    * (
                        -(jnp.cos(omega) * jnp.cos(Omega))
                        + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
                    )
                )
            )
            / (-1 + f1) ** 2
        )
    ) / r**2


def p_x0(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        2
        * a
        * (-1 + e**2)
        * (
            -(
                (
                    jnp.sin(f - theta)
                    * (
                        jnp.cos(omega)
                        * (
                            jnp.cos(Omega) * jnp.sin(theta)
                            + jnp.cos(i) * jnp.cos(theta) * jnp.sin(Omega)
                        )
                        + jnp.sin(omega)
                        * (
                            jnp.cos(theta) * jnp.cos(Omega)
                            - jnp.cos(i) * jnp.sin(theta) * jnp.sin(Omega)
                        )
                    )
                )
                / (-1 + f2) ** 2
            )
            + (
                jnp.cos(f - theta)
                * (
                    jnp.cos(theta)
                    * (2 - 2 * f1 + f1**2 + (-2 + f1) * f1 * jnp.cos(2 * phi))
                    * (
                        jnp.cos(omega) * jnp.cos(Omega)
                        - jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
                    )
                    - 2
                    * (
                        (-2 + f1)
                        * f1
                        * jnp.cos(phi)
                        * jnp.sin(i)
                        * jnp.sin(phi)
                        * jnp.sin(Omega)
                        + (-1 + f1) ** 2
                        * jnp.cos(phi) ** 2
                        * jnp.sin(theta)
                        * (
                            jnp.cos(Omega) * jnp.sin(omega)
                            + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
                        )
                        + jnp.sin(theta)
                        * jnp.sin(phi) ** 2
                        * (
                            jnp.cos(Omega) * jnp.sin(omega)
                            + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
                        )
                    )
                )
            )
            / (2.0 * (-1 + f1) ** 2)
        )
    ) / (r**2 * (1 + e * jnp.cos(f)))


def p_yy(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        (
            jnp.cos(Omega)
            * (
                jnp.sin(i) * jnp.sin(phi)
                + jnp.cos(i) * jnp.cos(phi) * jnp.sin(theta + omega)
            )
            + jnp.cos(phi) * jnp.cos(theta + omega) * jnp.sin(Omega)
        )
        ** 2
        + (
            jnp.cos(i) * jnp.cos(theta + omega) * jnp.cos(Omega)
            - jnp.sin(theta + omega) * jnp.sin(Omega)
        )
        ** 2
        / (-1 + f2) ** 2
        + (
            jnp.cos(phi) * jnp.cos(Omega) * jnp.sin(i)
            - jnp.sin(phi)
            * (
                jnp.cos(i) * jnp.cos(Omega) * jnp.sin(theta + omega)
                + jnp.cos(theta + omega) * jnp.sin(Omega)
            )
        )
        ** 2
        / (-1 + f1) ** 2
    ) / r**2


def p_yz(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        2
        * (
            -(
                (
                    jnp.cos(i) * jnp.sin(phi)
                    - jnp.cos(phi) * jnp.sin(i) * jnp.sin(theta + omega)
                )
                * (
                    jnp.cos(Omega)
                    * (
                        jnp.sin(i) * jnp.sin(phi)
                        + jnp.cos(i) * jnp.cos(phi) * jnp.sin(theta + omega)
                    )
                    + jnp.cos(phi) * jnp.cos(theta + omega) * jnp.sin(Omega)
                )
            )
            + (
                jnp.cos(theta + omega)
                * jnp.sin(i)
                * (
                    jnp.cos(i) * jnp.cos(theta + omega) * jnp.cos(Omega)
                    - jnp.sin(theta + omega) * jnp.sin(Omega)
                )
            )
            / (-1 + f2) ** 2
            + (
                (
                    jnp.cos(i) * jnp.cos(phi)
                    + jnp.sin(i) * jnp.sin(phi) * jnp.sin(theta + omega)
                )
                * (
                    -(jnp.cos(phi) * jnp.cos(Omega) * jnp.sin(i))
                    + jnp.sin(phi)
                    * (
                        jnp.cos(i) * jnp.cos(Omega) * jnp.sin(theta + omega)
                        + jnp.cos(theta + omega) * jnp.sin(Omega)
                    )
                )
            )
            / (-1 + f1) ** 2
        )
    ) / r**2


def p_y0(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        2
        * a
        * (-1 + e**2)
        * (
            jnp.cos(f - theta)
            * jnp.cos(phi)
            * (
                jnp.cos(Omega)
                * (
                    jnp.sin(i) * jnp.sin(phi)
                    + jnp.cos(i) * jnp.cos(phi) * jnp.sin(theta + omega)
                )
                + jnp.cos(phi) * jnp.cos(theta + omega) * jnp.sin(Omega)
            )
            + (
                jnp.sin(f - theta)
                * (
                    jnp.cos(i) * jnp.cos(theta + omega) * jnp.cos(Omega)
                    - jnp.sin(theta + omega) * jnp.sin(Omega)
                )
            )
            / (-1 + f2) ** 2
            + (
                jnp.cos(f - theta)
                * jnp.sin(phi)
                * (
                    -(jnp.cos(phi) * jnp.cos(Omega) * jnp.sin(i))
                    + jnp.sin(phi)
                    * (
                        jnp.cos(i) * jnp.cos(Omega) * jnp.sin(theta + omega)
                        + jnp.cos(theta + omega) * jnp.sin(Omega)
                    )
                )
            )
            / (-1 + f1) ** 2
        )
    ) / (r**2 * (1 + e * jnp.cos(f)))


def p_zz(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        (jnp.cos(theta + omega) ** 2 * jnp.sin(i) ** 2) / (-1 + f2) ** 2
        + (
            jnp.cos(i) * jnp.sin(phi)
            - jnp.cos(phi) * jnp.sin(i) * jnp.sin(theta + omega)
        )
        ** 2
        + (
            jnp.cos(i) * jnp.cos(phi)
            + jnp.sin(i) * jnp.sin(phi) * jnp.sin(theta + omega)
        )
        ** 2
        / (-1 + f1) ** 2
    ) / r**2


def p_z0(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        2
        * a
        * (-1 + e**2)
        * (
            (jnp.cos(theta + omega) * jnp.sin(i) * jnp.sin(f - theta))
            / (-1 + f2) ** 2
            + (
                jnp.cos(f - theta)
                * (
                    -(
                        (-2 + f1)
                        * f1
                        * jnp.cos(i)
                        * jnp.cos(phi)
                        * jnp.sin(phi)
                    )
                    + jnp.sin(i)
                    * ((-1 + f1) ** 2 * jnp.cos(phi) ** 2 + jnp.sin(phi) ** 2)
                    * jnp.sin(theta + omega)
                )
            )
            / (-1 + f1) ** 2
        )
    ) / (r**2 * (1 + e * jnp.cos(f)))


def p_00(a, e, f, Omega, i, omega, r, phi, theta, f1, f2):
    return (
        a**2
        * (-1 + e**2) ** 2
        * (
            jnp.sin(f - theta) ** 2 / (-1 + f2) ** 2
            + (
                jnp.cos(f - theta) ** 2
                * ((-1 + f1) ** 2 * jnp.cos(phi) ** 2 + jnp.sin(phi) ** 2)
            )
            / (-1 + f1) ** 2
        )
    ) / (r + e * r * jnp.cos(f)) ** 2


@jax.jit
def planet_3d_coeffs(a, e, f, Omega, i, omega, r, phi, theta, f1, f2, **kwargs):
    return {
        "p_xx": p_xx(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_xy": p_xy(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_xz": p_xz(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_x0": p_x0(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_yy": p_yy(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_yz": p_yz(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_y0": p_y0(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_zz": p_zz(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_z0": p_z0(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
        "p_00": p_00(a, e, f, Omega, i, omega, r, phi, theta, f1, f2),
    }
