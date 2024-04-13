# %%
from scipy.interpolate import RegularGridInterpolator
import numpy as np

# %% AGMA Bending Geometry Facotr J (Table 12-9 of textbook)
def J_factor_pinion_template(Ng, Np):   
    #   Data from the table
    Ng_data = np.array([55, 135])
    Np_data = np.array([21, 26, 35, 55])
    J_factor_data = np.array([[0.34, 0.37, 0.40, 0.43], [0.35, 0.38, 0.41, 0.45]])
    # Interpolates the data from the table
    interp = RegularGridInterpolator((Ng_data, Np_data), J_factor_data)
    return interp([Ng, Np])[0]

# Vectorizes the function so that it takes mulltidimensional inputs
J_pinion = np.vectorize(J_factor_pinion_template)

def J_factor_gear_template(Ng, Np):    
    # Data from the table
    Ng_data = np.array([55, 135])
    Np_data = np.array([21, 26, 35, 55])
    J_factor_data = np.array([[0.40, 0.41, 0.42, 0.43], [0.43, 0.44, 0.45, 0.47]])
    # Interpolates the data from the table
    interp = RegularGridInterpolator((Ng_data, Np_data), J_factor_data)
    return interp([Ng, Np])[0]

# Vectorizes the function so that it takes mulltidimensional inputs
J_gear = np.vectorize(J_factor_gear_template)

# %% Load Distribution Factor (Table 12-16 of textbook)
def Km_factor_template(face_width):
    if face_width < 2:
            return 1.6
    elif face_width < 6:
            return 1.6 + (1.7-1.6)/(6-2)*(face_width-2)
    elif face_width < 9:
        return 1.7 + (1.8-1.7)/(9-6)*(face_width-6)
    elif face_width < 20:
        return 1.8 + (2-1.8)/(20-9)*(face_width-9)
    else:
        return 2

# Vectorizes the function so that it takes mulltidimensional inputs
Km = np.vectorize(Km_factor_template)  
Cm = Km # The function for surface fatigue is the same as bending fatigue

# %% Dynamic Factor (Equation 12.19us from the textbook)
def Kv_factor_template(tangential_velocity):
    Qv = 11 # Assuming a quality factor of 11 for aerospace applications
    B = (12 - Qv)**(2/3) / 4
    A = 50 + 56*(1 - B)
    return (A / (A + np.sqrt(tangential_velocity)))**B

# Vectorizes the function so that it takes mulltidimensional inputs
Kv = np.vectorize(Kv_factor_template)
Cv = Kv # The function for surface fatigue is the same as bending fatigue

# %% Bending Stress Equation (Equaton 12.15us)
def bending_stress(nb_pinion_teeth, nb_gear_teeth, face_width, torque, 
                        rotational_speed, diametral_pitch = 10, Ka = 1.12, 
                        Ks = 1, Kb = 1, pinion = False, idler = False):
    
    Np_mesh, fw_mesh, = np.meshgrid(nb_pinion_teeth, face_width,  indexing='ij')
    Ng_mesh, fw_mesh = np.meshgrid(nb_gear_teeth, face_width, indexing='ij')

    # Uses different bending factor for pinion and gear
    if pinion:
        d_mesh = Np_mesh / diametral_pitch
        J_mesh = J_pinion(Ng_mesh, Np_mesh)
    else:
        d_mesh = Ng_mesh / diametral_pitch
        J_mesh = J_gear(Ng_mesh, Np_mesh)
    
    # Uses different idler factor if gear is idler or not
    if idler:
        Ki = 1.42
        Wt_mesh = torque / d_mesh
    else:
        Ki = 1
        Wt_mesh = torque / (d_mesh / 2)
    
    # Computes the pitch-line velocity and the dynamic factor
    Vt_mesh = rotational_speed * (d_mesh / 2) * 60 / 12
    Kv_mesh = Kv(Vt_mesh)

    # Computes load distribution factor
    Km_mesh = Km(fw_mesh)

    return (Wt_mesh * diametral_pitch * Ka * Km_mesh * Ks * Kb * Ki)  / (J_mesh * fw_mesh * Kv_mesh)

# %% Surface Stress Equation (Equaton 12.21)
def surface_stress(nb_pinion_teeth, nb_gear_teeth, face_width, torque, 
                        rotational_speed, diametral_pitch = 10, Ca = 1.12, 
                        Cs = 1, Cp = 2300 ,pinion = False, idler = False):
    
    Np_mesh, fw_mesh, = np.meshgrid(nb_pinion_teeth, face_width,  indexing='ij')
    Ng_mesh, fw_mesh = np.meshgrid(nb_gear_teeth, face_width, indexing='ij')

    # Diameter computed depending on if it is a gear or a pinion
    if pinion:
        d_mesh = Np_mesh / diametral_pitch
        dp_mesh = d_mesh
        dg_mesh = Ng_mesh / diametral_pitch
    else:
        d_mesh = Ng_mesh / diametral_pitch
        dg_mesh = d_mesh
        dp_mesh = Np_mesh / diametral_pitch

    # Radius of curvature is computed
    rho_p = np.sqrt((dp_mesh / 2 + 1 / diametral_pitch)**2 - (dp_mesh * np.cos(np.deg2rad(20)) / 2)**2) - np.pi * np.cos(np.deg2rad(20)) / diametral_pitch
    C = (dp_mesh + dg_mesh) / 2
    rho_g = C * np.sin(np.deg2rad(20)) - rho_p

    # Computes surface geometry factor
    I_mesh = np.cos(np.deg2rad(20)) / (1 / rho_p + 1 / rho_g) / dp_mesh

    # Computes the tangential force from the applied torque depending on if its 
    # a pinion or a gear
    if idler:
        Wt_mesh = torque / d_mesh
    else:
        Wt_mesh = torque / (d_mesh / 2)
    
    # Computes pitch-line velocity
    Vt_mesh = rotational_speed * (d_mesh / 2) * 60 / 12
    # Computes dynamic factor
    Cv_mesh = Cv(Vt_mesh)
    # Computes load distribution factor
    Cm_mesh = Cm(fw_mesh)

    return Cp * np.sqrt((Wt_mesh * Ca * Cm_mesh * Cs) / (fw_mesh * I_mesh * d_mesh * Cv_mesh))

# %% Objective function of the optimization problem (see section in report for mor details)
# Computes the volume of all gears
def total_gear_volume(Np, Ng, fw1, fw2, diametral_pitch = 10):
    Np_mesh, fw1_mesh, fw2_mesh = np.meshgrid(Np, fw1, fw2, indexing='ij')
    Ng_mesh, fw1_mesh, fw2_mesh = np.meshgrid(Ng, fw1, fw2, indexing='ij')
    dg_mesh = Ng_mesh / diametral_pitch
    dp_mesh = Np_mesh / diametral_pitch
    return np.pi / 4 * (np.power(dp_mesh, 2) * (fw1_mesh + 2 * fw2_mesh) + np.power(dg_mesh, 2) * (2 * fw1_mesh + fw2_mesh))