import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# =====================================================
# Load .nc file
# =====================================================

filename = r"C:\Users\Rodrigo\Desktop\Rodrigo\02 - Research fellow\01 - Storm surge ML downscaling\Experiments_v6_v2\surge-ml_v2 - sliding-window\data\coarse data\med-mfc\medmfc_ssh.nc"

assert os.path.exists(filename), "File not found. Check path."

ds = xr.open_dataset(filename)

print("Dataset structure:")
print(ds)

# =====================================================
# Extract variables
# =====================================================

par = ds["ssh"].values
lat = ds["lat"].values
lon = ds["lon"].values
tnum = ds["time"].values.astype(float)

# =====================================================
# FIX DIMENSION ORDER (CRITICAL)
# Ensure (lon, lat, time) 
# =====================================================

dims = ds["ssh"].dims
print("Original dimensions:", dims)

if dims == ('time', 'lat', 'lon'):
    par = np.transpose(par, (2, 1, 0))  # → (lon, lat, time)

elif dims == ('lat', 'lon', 'time'):
    par = np.transpose(par, (1, 0, 2))  # → (lon, lat, time)

elif dims == ('lon', 'lat', 'time'):
    pass  # already correct

else:
    raise ValueError(f"Unexpected dimension order: {dims}")

# =====================================================
# Create grid
# =====================================================

LON, LAT = np.meshgrid(lon, lat, indexing="ij")

# =====================================================
# Time conversion 
# =====================================================

tvec = np.array([datetime(1970,1,1) + timedelta(days=float(t)) for t in tnum])

# =====================================================
# Preprocessing for PCA
# =====================================================

m, n, p = par.shape  # lon, lat, time

# Reshape: (space, time) → transpose → (time, space)
tofill = par.reshape(m*n, p).T

# Replace NaNs with 0 
tofill0 = np.copy(tofill)
tofill0[np.isnan(tofill0)] = 0

# =====================================================
# PCA
# =====================================================

pca = PCA()
score = pca.fit_transform(tofill0)        # (time, components)
coeff = pca.components_.T                # (space, components)
expl = pca.explained_variance_ratio_ * 100

print(f"\nPC1 variance explained: {expl[0]:.2f}%")

# =====================================================
# Select number of components
# =====================================================

idx_cm = 7
csvar = np.cumsum(expl)

# =====================================================
# Plot explained variance
# =====================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.bar(range(1, len(expl)+1), expl)
plt.xlabel("Principal components")
plt.ylabel("Variance explained [%]")
plt.grid()

plt.subplot(1,2,2)
plt.plot(range(1, len(expl)+1), csvar, "-")
plt.xlabel("Principal components")
plt.ylabel("Cumulative variance explained [%]")
plt.grid()

plt.tight_layout()
plt.show()

# =====================================================
# Extract time series at Trieste
# =====================================================

pred_coord = np.array([45.68, 13.25])  # lat, lon

# Find closest grid point
idx_lat = np.argmin(np.abs(lat - pred_coord[0]))
idx_lon = np.argmin(np.abs(lon - pred_coord[1]))

print(f"Selected grid point → lat: {lat[idx_lat]}, lon: {lon[idx_lon]}")

# =====================================================
# Reconstruct PCA time series 
# =====================================================

srs_fill = np.zeros((p, idx_cm))

for j in range(idx_cm):
    spatial_mode = coeff[:, j].reshape(m, n)

    # Weight at location
    weight = spatial_mode[idx_lon, idx_lat]

    # Time series
    srs_fill[:, j] = score[:, j] * weight

# =====================================================
# Build output 
# =====================================================

years   = np.array([d.year for d in tvec])
months  = np.array([d.month for d in tvec])
days    = np.array([d.day for d in tvec])
hours   = np.array([d.hour for d in tvec])
minutes = np.array([d.minute for d in tvec])
seconds = np.array([d.second for d in tvec])

serie = np.column_stack([
    years, months, days,
    hours, minutes, seconds,
    tnum,
    srs_fill
])

# =====================================================
# Save output
# =====================================================

np.savetxt("ssh.txt", serie, fmt="%.6f")

print("\nPCA completed successfully.")
print("Output saved as ssh.txt")