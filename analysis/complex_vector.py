import numpy as np
import matplotlib.pyplot as plt

# Example sea-ice velocity components
u_i_t = 3.0   # zonal component
v_i_t = 2.0   # meridional component

# Complex vector
z_i_t = u_i_t + 1j * v_i_t

fig, ax = plt.subplots(figsize=(4, 4))

# Plot vector from origin to (u, v)
ax.quiver(
    0, 0,
    z_i_t.real, z_i_t.imag,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.01
)

# Component guide lines
ax.plot([z_i_t.real, z_i_t.real], [0, z_i_t.imag], "k--", linewidth=1)
ax.plot([0, z_i_t.real], [z_i_t.imag, z_i_t.imag], "k--", linewidth=1)


ax.text(
    0.90, 0.82,
    r"$z_{i,t} = u_{i,t} + i v_{i,t}$",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=12
)

ax.set_xlabel(r"Re: $u_{i,t}$")
ax.set_ylabel(r"Im: $v_{i,t}$")

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("complex_velocity_vector.png", dpi=300, bbox_inches="tight")
plt.show()