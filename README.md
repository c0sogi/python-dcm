# python-dcm

A high-performance Python package for handling ETAS DCM(Data Conversion Format) files used in engine calibration tools like INCA, MDA, EHANDBOOK, and CANape.

## Quick Start

```python
from dcm import DCM

# Read a DCM file
dcm = DCM.from_file('calibration.dcm')

# Access calibration data
parameter_value = dcm.parameters['ENGINE_SPEED'].value
map_data = dcm.maps['FUEL_MAP'].dataframe
curve_data = dcm.curves['BOOST_CURVE'].series

# Interpolate values
x_points = [1000, 1500, 2000]  # RPM
y_points = [50, 75, 100]       # Load %
interpolated = dcm.maps['FUEL_MAP'].as_function(x_points, y_points)

# Visualize data
dcm.maps['FUEL_MAP'].to_figure()
```

## Key Features

- **Easy Data Access**: Directly access parameters, curves, and maps with Pandas integration
- **Interpolation**: Built-in 1D/2D linear interpolation for real-time value calculation
- **Visualization**: One-line plotting of characteristic curves and maps
- **Excel Integration**: Import/export calibration data from Excel spreadsheets
- **Set Operations**: Compare and merge DCM files with `|`, `-`, `&`, and `%` operators
- **Type Support**: Handle all DCM data types including fixed/group characteristics

## Installation

Requires Python ≥ 3.10

```bash
pip install python-dcm
```

## Common Use Cases

### Data Access & Manipulation
```python
# Get parameter value
rpm_limit = dcm.parameters['MAX_RPM'].value

# Access map as DataFrame
fuel_map = dcm.maps['FUEL_MAP'].dataframe
fuel_map.iloc[0, 0] = 14.7  # Modify value

# Get curve data
boost_curve = dcm.curves['BOOST_CURVE'].series
max_boost = boost_curve.max()
```

### Excel Integration
```python
# Load calibration data from Excel
dcm.load_from_excel(
    maps_path='maps.xlsx',
    curves_path='curves.xlsx',
    parameters_path='params.xlsx'
)

# Each sheet name becomes the calibration object name
```

### Visualization
```python
import matplotlib.pyplot as plt

# Plot a map with custom settings
fig, ax = dcm.maps['FUEL_MAP'].to_figure(
    cmap='viridis',
    fontsize=12
)
plt.show()

# Plot multiple curves
fig, ax = plt.subplots()
dcm.curves['BOOST_LOW'].to_figure(ax=ax, label='Low')
dcm.curves['BOOST_HIGH'].to_figure(ax=ax, label='High')
plt.legend()
```

### Advanced Features

#### Parameter Type Conversion
```python
param = dcm.parameters['CONTROL_BITS']
binary = param.as_bin()    # [1, 3, 5] (bits set to 1)
hex_val = param.as_hex()   # [A, F, 1] (hexadecimal digits)
```

#### DCM File Comparison
```python
# Find differences between calibrations
modified = old_dcm % new_dcm
print(modified.parameters.keys())  # Changed parameters

# Merge calibrations
combined = dcm1 | dcm2
```

## Supported Data Types

- Parameters (FESTWERT)
- Parameter Blocks (FESTWERTEBLOCK)
- Characteristic Lines (KENNLINIE/FESTKENNLINIE/GRUPPENKENNLINIE)
- Characteristic Maps (KENNFELD/FESTKENNFELD/GRUPPENKENNFELD)
- Distributions (STUETZSTELLENVERTEILUNG)
- Text Strings (TEXTSTRING)

## Dependencies

- NumPy ≥ 1.20.0
- Pandas ≥ 1.5.0
- Matplotlib ≥ 3.0.0
- OpenPyXL ≥ 3.1.0

## License

MIT License

## Contributing

Contributions welcome! Please format code with [ruff](https://docs.astral.sh/ruff/) before submitting PRs.

## Contact

- Author: c0sogi
- Email: dcas@naver.com or cosogi1@gmail.com

Feel free to reach out for questions or suggestions.

---

For detailed documentation and examples, visit our [GitHub repository](https://github.com/c0sogi/python-dcm).
