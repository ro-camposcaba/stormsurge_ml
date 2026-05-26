
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfes
import os
import yaml


# Absolute path to FES2014 root folder
with open('config_tide.yml', 'r') as f:
    config = yaml.safe_load(f)

fes_root = config['fes_root']
lon = config['coordinates']['lon']
lat = config['coordinates']['lat']
start_date = np.datetime64(config['time_range']['start'])
end_date = np.datetime64(config['time_range']['end'])
output_file = config['output']['file']
num_threads = config.get('num_threads', 1)

# Configuration file on FES2014 main folder, which contains tide constituents
config_file = os.path.join(fes_root, "fes_slev.yml")

# Save current working directory
original_cwd = os.getcwd()

# Temporarily change working directory
os.chdir(fes_root)

# Load the config 
handlers = pyfes.load_config(config_file)

# Go back to original working directory
os.chdir(original_cwd)


# Generate hourly timestamps
dates = np.arange(start_date, end_date + np.timedelta64(1, 'h'), np.timedelta64(1, 'h'))

# Truncate times to the nearest hour (optional if already aligned)
dates = dates.astype('datetime64[h]')


lons = np.full(dates.shape, lon)
lats = np.full(dates.shape, lat)

tide, lp, _ = pyfes.evaluate_tide(handlers['tide'],
                                  dates,
                                  lons,
                                  lats,
                                  num_threads=1)

cnes_julian_days = (dates - np.datetime64('1950-01-01T00:00:00')
                    ).astype('M8[s]').astype(float) / 86400
hours = cnes_julian_days % 1 * 24


# Convert tide from cm to meters
tide_meters = tide / 100
# Compute the mean tide level
mean_tide = np.mean(tide_meters)
print(f"Mean Tide Level: {mean_tide:.3f} m")

# Plot the tide time-series
plt.figure(figsize=(12, 5))
plt.plot(dates, tide_meters, label="Tide Level (m)", color="b", linewidth=1)
plt.axhline(mean_tide, color="r", linestyle="--", label=f"Mean Tide: {mean_tide:.3f} m")

# Formatting the plot
plt.xlabel("Time")
plt.ylabel("Tide Level (m)")
plt.title("Tide Time-Series")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# Extract components directly from datetime64 array
years = dates.astype('datetime64[Y]').astype(int) + 1970  # NumPy year starts at 1970
months = (dates.astype('datetime64[M]') - dates.astype('datetime64[Y]')).astype(int) + 1
days = (dates.astype('datetime64[D]') - dates.astype('datetime64[M]')).astype(int) + 1
hours = (dates.astype('datetime64[h]') - dates.astype('datetime64[D]')).astype(int)

# Set minutes and seconds explicitly to 0
minutes = np.zeros_like(hours)
seconds = np.zeros_like(hours)

# Create DataFrame
df = pd.DataFrame({
    'Year': years,
    'Month': months,
    'Day': days,
    'Hour': hours,
    'Minute': minutes,
    'Second': seconds,
    'Julian Date': cnes_julian_days,
    'Tide (m)': tide_meters
})

# Save to .txt file
df.to_csv(output_file, sep='\t', index=False, header = False)

print(f"Data saved to {output_file}")