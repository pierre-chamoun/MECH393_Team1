# Bending fatigue strength parameters
mpa2psi = 145.038
bending_fatigue_strength = 1264 * mpa2psi  # psi

# Density
density = 0.2880 # lb/in3

# Hardness
brinell_hardness = 695  # HB

# Surface fatigue strength parameters
Ct, Cr, Ch = 1, 1, 1
Cl = 2.466 * (34426849315)**(-0.056)
uncorrecterd_surface_fatigue_strength = 349 * brinell_hardness + 34300                    # psi
surface_fatigue_strength = (Cl * Ch) / (Ct * Cr) * uncorrecterd_surface_fatigue_strength  # psi   



