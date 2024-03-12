import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bulge.engine.planet_3d import planet_3d_coeffs
from bulge.engine.planet_2d import planet_2d_coeffs
from bulge.engine.parametric_ellipse import poly_to_parametric

class System:
    def __init__(self, params):
        state_keys = [
            "a",
            "e",
            "i",
            "Omega",
            "omega",
            "f",
            "phi",
            "theta",
            "r",
            "f1",
            "f2",
            "u1",
            "u2",
        ]

        state = {}
        for key in state_keys:
            if key in params.keys():
                state[key] = params[key]
            else:
                if key == "a":
                    raise ValueError(
                        "Semi-major axis 'a' is a required parameter."
                    )
                elif key == "e":
                    state["e"] = 0.0
                elif key == "i":
                    state["i"] = jnp.pi / 2
                elif key == "Omega":
                    state["Omega"] = jnp.pi
                elif key == "omega":
                    state["omega"] = 0.0
                elif key == "f":
                    state["f"] = jnp.pi / 2
                elif key == "phi":
                    state["phi"] = 0.0
                elif key == "theta":
                    state["theta"] = 0.0
                elif key == "r":
                    raise ValueError(
                        "Equitorial radius 'r' is a required parameter."
                    )
                elif key == "f1":
                    state["f1"] = 0.0
                elif key == "f2":
                    state["f2"] = 0.0
                elif key == "u1":
                    state["u1"] = 0.6
                elif key == "u2":
                    state["u2"] = 0.2

        # make sure the user provided no keys that are not in state_keys
        for key in params.keys():
            if key not in state_keys and (key != "q1" | key != "q2"):
                raise ValueError(f"Invalid parameter '{key}'")

        if state["e"] == 0:
            assert state["omega"] == 0, "omega must be 0 for a circular orbit"

        if "q1" in params.keys() or "q2" in params.keys():
            if not ("q1" in params.keys() and "q2" in params.keys()):
                raise ValueError(
                    "Provide both 'q1' and 'q2' for quadratic limb darkening."
                )
            if "u1" in params.keys() or "u2" in params.keys():
                raise ValueError(
                    "Provide either 'q1' and 'q2' or 'u1' and 'u2', not both."
                )
            state["u1"] = 2 * jnp.sqrt(params["q1"]) * params["q2"]
            state["u2"] = jnp.sqrt(params["q1"]) * (1 - 2 * params["q2"])


        self._state = state
        self._coeffs_3d = planet_3d_coeffs(**self._state)
        self._coeffs_2d = planet_2d_coeffs(**self._coeffs_3d)
        self._para_coeffs_2d = poly_to_parametric(**self._coeffs_2d)

