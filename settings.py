import numpy as np

# Coordinate range for shot locations given by data publisher
XMIN = -250
XMAX = 250
YMIN = -52
YMAX = 420

OFFSET = 1e-5
# Plot Settings
# Bin counts for 3d bar plot
XBINS = 61
YBINS = 122
denom = 1 if XBINS*YBINS==1 else XBINS*YBINS-1
k = -np.log(0.005)/(denom)
CLIP_RANGE = np.e**(-k*(XBINS*YBINS-1))
ALPHA_BAR = 0.5
ALPHA_OVERLAY = 0.85
CMAP = 'viridis'
CMAP_EXP = 'magma'
SHRINK = 1
