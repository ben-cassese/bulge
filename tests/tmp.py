# rkey = jax.random.PRNGKey(10)
# for i in tqdm(range(1000)):
#     rkey, *subkeys = jax.random.split(rkey, num=11)

#     ugh = False
#     for f in jnp.linspace(0, 2*jnp.pi, 100):
#         f1 = jax.random.uniform(subkeys[5], minval=0.0, maxval=0.8)
#         state = {
#             "a" : jax.random.uniform(subkeys[0], minval=1.1, maxval=300),
#             "e" : jax.random.uniform(subkeys[1], minval=0.0, maxval=0.99),
#             "i" : jax.random.uniform(subkeys[2], minval=0.0, maxval=jnp.pi),
#             "Omega" : jax.random.uniform(subkeys[3], minval=0.0, maxval=2*jnp.pi),
#             "omega" : jax.random.uniform(subkeys[4], minval=0.0, maxval=2*jnp.pi),
#             "f" : f,
#             "f1" : f1,
#             "f2" : jax.random.uniform(subkeys[6], minval=0.0, maxval=f1),
#             "r" : jax.random.uniform(subkeys[7], minval=0.0, maxval=1.0),
#             "phi" : jax.random.uniform(subkeys[8], minval=0.0, maxval=jnp.pi),
#             "theta" : jax.random.uniform(subkeys[9], minval=0.0, maxval=2*jnp.pi),
#         }
#         s = System(state)
#         c = poly_to_parametric(**s._coeffs_2d)
#         for key in c.keys():
#             if jnp.isnan(c[key]):
#                 print(f)
#                 print(c)
#                 ugh = True
#         if ugh: break



# rkey = jax.random.PRNGKey(13)

# for i in range(10):
#     rkey, *subkeys = jax.random.split(rkey, num=12)

#     ugh = False
#     f1 = jax.random.uniform(subkeys[5], minval=0.0, maxval=0.8)
#     r = jax.random.uniform(subkeys[7], minval=0.0, maxval=1.0)
#     state = {
#         "a" : jax.random.uniform(subkeys[0], minval=1.1, maxval=300),
#         "e" : jax.random.uniform(subkeys[1], minval=0.0, maxval=0.99),
#         "i" : jax.random.uniform(subkeys[2], minval=0.0, maxval=jnp.pi),
#         "Omega" : jax.random.uniform(subkeys[3], minval=0.0, maxval=2*jnp.pi),
#         "omega" : jax.random.uniform(subkeys[4], minval=0.0, maxval=2*jnp.pi),
#         "f" : jax.random.uniform(subkeys[10], minval=0.0, maxval=2*jnp.pi),
#         "f1" : f1,
#         "f2" : jax.random.uniform(subkeys[6], minval=0.0, maxval=f1),
#         "r" : r,
#         "phi" : jax.random.uniform(subkeys[8], minval=0.0, maxval=jnp.pi),
#         "theta" : jax.random.uniform(subkeys[9], minval=0.0, maxval=2*jnp.pi),
#     }
#     s = System(state)
#     c = poly_to_parametric(**s._coeffs_2d)

#     fig, ax = plt.subplots()
#     x = jnp.linspace(c["c_x3"] - r, c["c_x3"] + r, 1000)
#     y = jnp.linspace(c["c_y3"] - r, c["c_y3"] + r, 1000)
#     X, Y = jnp.meshgrid(x, y)
#     planet = (
#         s._coeffs_2d["rho_xx"] * X**2
#         + s._coeffs_2d["rho_yy"] * Y**2
#         + s._coeffs_2d["rho_xy"] * X * Y
#         + s._coeffs_2d["rho_x0"] * X
#         + s._coeffs_2d["rho_y0"] * Y
#         + s._coeffs_2d["rho_00"]
#     )

#     ax.contour(X, Y, planet, levels=[1.0], colors="red", linewidths=10)

#     parametric_angle = jnp.linspace(0, 2*jnp.pi, 1000)
#     x_vals = c["c_x1"] * jnp.cos(parametric_angle) + c["c_x2"] * jnp.sin(parametric_angle) + c["c_x3"]
#     y_vals = c["c_y1"] * jnp.cos(parametric_angle) + c["c_y2"] * jnp.sin(parametric_angle) + c["c_y3"]
#     ax.plot(x_vals, y_vals, color="blue", linewidth=1)

#     ax.set(xlim=(c["c_x3"] - r, c["c_x3"] + r), ylim=(c["c_y3"] - r, c["c_y3"] + r), aspect="equal")