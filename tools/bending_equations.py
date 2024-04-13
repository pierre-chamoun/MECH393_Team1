# %%
import numpy as np

# %% 
#Static concentration factor for bending (Figure 27 in report)
def Kt_bending_step(D, d, r):
    # Data in table
    A_values = np.array([0.91938, 0.96048, 0.98061, 0.98137, 0.97527, 0.95120, 
                         0.97098, 0.93836, 0.90879, 0.89334, 0.87868])
    b_values = -np.array([0.17032, 0.17711, 0.18381, 0.19653, 0.20958, 0.23757,
                           0.21796, 0.26759, 0.28598, 0.30860, 0.33243])
    diameter_ratio = np.array([1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.2, 1.5, 2, 3, 6])
    # Interpolation of the data for points in between
    A = np.interp(D/d, diameter_ratio, A_values)
    b = np.interp(D/d, diameter_ratio, b_values)
    return A * (r/d) ** b

# Static concentration factor for torsion (Figure  28 in report)
def Kt_torsion_step(D, d, r):
    # Data in table
    A_values = np.array([0.90337, 0.83425, 0.84897, 0.86331])
    b_values = -np.array([0.12692, 0.21649, 0.23161, 0.23865])
    diameter_ratio = np.array([1.09, 1.20, 1.33, 2.00])
    # Interpolation of the data for points in between
    A = np.interp(D/d, diameter_ratio, A_values)
    b = np.interp(D/d, diameter_ratio, b_values)
    return A * (r/d) ** b

# %%
# Static stress concentration of a key (Figure 29 of report)
def Kt_key (d, r):
    # The graph is digitized to obtain a function that represents the graph
    Kt_data = np.loadtxt("kt_key_data.csv", delimiter=",")[:,1]
    notch_ratio_data = np.loadtxt("kt_key_data.csv", delimiter=",")[:,0]
    Kt = np.interp(r/d, notch_ratio_data, Kt_data)
    return Kt
# Static stress concentration of a key for torsion (Figure 29 of the report)
def Kts_key (d, r):
    # The graph is digitized to obtain a function that represents the graph
    Kts_data = np.loadtxt("kts_key_data.csv", delimiter=",")[:,1]
    notch_ratio_data = np.loadtxt("kts_key_data.csv", delimiter=",")[:,0]
    Kts = np.interp(r/d, notch_ratio_data, Kts_data)
    return Kts

# %%
# Neubeur constant (Figure 30 of report)
def neuber_constant(Sut):
    # Data in table
    sut_values = np.array([50,55,60,70,80,90,100,110,120,130,140,160,180,200,220,240])
    neuber_values = np.array([0.130,0.118,0.108,0.093,0.080,0.070,0.062,0.055,
                              0.049,0.044,0.039,0.031,0.024,0.018,0.013,0.009])

    return np.interp(Sut,sut_values,neuber_values) # Return interpolation

# %%
# Computes the fatigue stress concentration for bending for a step (Equation 6.11 of textbook)
def Kf_step(d, D, r=0.1, Sut=65000):
        
        neuber_bending = neuber_constant(Sut * 1E-3)      # Computes neubeur constant
        q_bending = 1 / (1 + neuber_bending / np.sqrt(r)) # Computes notch sensitivity
        Kt = Kt_bending_step(D, d, r) # Computes static stress concentration
        Kf = 1 + q_bending * (Kt - 1) # Computes fatigue stress concentration

        return Kf

# Computes the fatigue stress concentration for torsion for a step (Equation 6.11 of textbook)
def Kfs_step(d, D, r=0.1, Sut=65000):

        neuber_torsion = neuber_constant((Sut + 20000) * 1E-3)  # Computes neubeur constant
        q_torsion = 1 / (1 + neuber_torsion / np.sqrt(r))       # Computes notch sensitivity
        Kts = Kt_torsion_step(D, d, r)  # Computes static stress concentration
        Kfs = 1 + q_torsion * (Kts - 1) # Computes fatigue stress concentration

        return Kfs

# %%
# Computes the fatigue stress concentration for bending for a key (Equation 6.11 of textbook)
def Kf_key(d, r=0.01, Sut=65000):
        
        neuber_bending = neuber_constant(Sut * 1E-3)        # Computes neubeur constant
        q_bending = 1 / (1 + neuber_bending / np.sqrt(r))   # Computes notch sensitivity
        Kt = Kt_key(d, r)               # Computes static stress concentration
        Kf = 1 + q_bending * (Kt - 1)   # Computes fatigue stress concentration

        return Kf

# Computes the fatigue stress concentration for torsion for a key (Equation 6.11 of textbook)
def Kfs_key(d, r=0.01, Sut=65000):

        neuber_torsion = neuber_constant((Sut + 20000) * 1E-3)  # Computes neubeur constant
        q_torsion = 1 / (1 + neuber_torsion / np.sqrt(r))       # Computes notch sensitivity
        Kts = Kts_key(d, r)                 # Computes static stress concentration
        Kfs = 1 + q_torsion * (Kts - 1)     # Computes fatigue stress concentration

        return Kfs

#%%
# Computes the required shaft dimater (Equation 10.8 of textbook)
def shaft_diameter(Ma, Mm, Ta, Tm, Kf, Kfs, Sf, Sut = 65000, Nf = 1.6):
    
    Kfm = Kf    # Assumes mean stress concentration equal for conservative design
    Kfsm = Kfs  # Assumes mean stress concentration equal for conservative design
    Kfsm = Kfs

    return (32 * Nf / np.pi * (np.sqrt((Kf * Ma)**2 + 0.75*(Kfs * Ta)**2) 
                               / Sf + np.sqrt((Kfm * Mm)**2 + 0.75*(Kfsm * Tm)**2) / Sut)) ** (1/3)

# Computes the corrected fatigue strenght
def get_Sf(d, Ctemp=1, Cload =1, Csurf=0.89319, Crel=0.75, Se = 32500):
      
      Csize = 0.869 * d ** -0.097 # Computes Csize depending on the diameter

      return Ctemp * Csurf * Crel * Csize * Cload * Se