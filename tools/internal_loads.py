# %%
import numpy as np
from scipy import integrate

# %%
# Computes the raction forces for a simply supported beam
def reaction_forces(applied_loads, length_of_beam):
    """ Computes the raction forces for a simply supported beam.
        The user needs to provide a list of list containing the load as a 
        3d numpy array and the position as a 3d numpy array.  """
    net_applied_loads = np.zeros(6)

    for load, position in applied_loads:
        net_applied_loads += np.block([load, np.cross(position, load)])
    
    system_matrix = np.block([[np.eye(3), np.eye(3)], 
                              [np.zeros((3,3)), np.array([[0, -length_of_beam, 0], [length_of_beam, 0, 0], [0, 0, 0]])]])
    result = np.linalg.pinv(system_matrix) @ -net_applied_loads
    return [result[0:3], result[3:]]

# %%
# Computes the internal loads for a simply supported beam at n grid points
def internal_loads(loads, length_of_beam, nb_grid_points):
    """ Computes the internal forces for a simply supported beam.
    The user needs to provide a list of list containing the loads as a 
    3d numpy array and the position as a 3d numpy array.  """
    z = np.linspace(0, length_of_beam, nb_grid_points)
    reaction_1, reaction_2 = reaction_forces(loads, length_of_beam)
    
    # Computes the shear in the beam
    shears = np.zeros((2, nb_grid_points))
    shears[:, z > 0] += np.array([reaction_1[:2]]).T
    shears[:,z == length_of_beam] += np.array([reaction_2[:2]]).T
    for load, position in loads:
        shears[:, z >= position[2]] += np.array([load[:2]]).T

    # Integrates the shear numerically to obtain the bending moment
    moments = np.zeros((2, len(z)))
    for i in range(len(z)):
        if i == 0:
            continue
        else:
            moments[:, i] = integrate.trapezoid(shears[:, :i + 1], z[:i + 1])

    return z, shears, moments

