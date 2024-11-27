# DCM

A Python package for reading and writing DCM (Data Conservation Format) files used in various ECU calibration tools such as INCA, MDA, EHANDBOOK, CANape, and more.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Reading a DCM File](#reading-a-dcm-file)
  - [Writing a DCM File](#writing-a-dcm-file)
  - [Accessing Data](#accessing-data)
  - [Loading Data from Excel](#loading-data-from-excel)
  - [Interpolation Functions](#interpolation-functions)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Features

- **Read and Write DCM Files**: Parse and generate DCM files used in ECU calibration.
- **Data Manipulation**: Access and modify parameters, curves, maps, and other calibration data.
- **Excel Integration**: Load calibration data from Excel files.
- **Interpolation**: Perform 1D and 2D linear interpolation on calibration data.
- **Visualization**: Plot characteristic lines and maps using Matplotlib.
- **Support for Various Data Types**: Handle parameters, parameter blocks, characteristic lines, characteristic maps, distributions, and text strings.

## Installation

Ensure you have Python 3.11 or higher installed. Install the package using `pip`:

```bash
pip install git+https://github.com/c0sogi/python-dcm.git
```

Or if you have the package files locally:

```bash
pip install .
```

## Usage

### Reading a DCM File

```python
from dcm import DCM

# Read a DCM file
dcm = DCM.from_file('path/to/your_file.dcm')
```

### Writing a DCM File

```python
# Write the DCM object to a file
dcm.write('path/to/output_file.dcm')
```

### Accessing Data

```python
# Access a parameter
parameter = dcm.parameters['PARAMETER_NAME']
print(parameter.value)

# Access a characteristic line (curve)
curve = dcm.curves['CURVE_NAME']
print(curve.series)

# Access a characteristic map
char_map = dcm.maps['MAP_NAME']
print(char_map.dataframe)
```

### Loading Data from Excel

You can load calibration data from Excel files into the DCM object.

```python
# Load maps from an Excel file
dcm.load_maps('maps.xlsx')

# Load curves from an Excel file
dcm.load_lines('curves.xlsx')

# Load parameters from an Excel file
dcm.load_parameters('parameters.xlsx')

# Load parameter blocks from an Excel file
dcm.load_parameter_blocks('parameter_blocks.xlsx')
```

### Interpolation Functions

Perform interpolation using the calibration data.

#### 1D Interpolation (Characteristic Line)

```python
# Get the interpolation function for a curve
curve_function = curve.as_function

# Interpolate at specific points
import numpy as np
x_values = np.array([1.0, 2.0, 3.0])
interpolated_values = curve_function(x_values)
```

#### 2D Interpolation (Characteristic Map)

```python
# Get the interpolation function for a map
map_function = char_map.as_function

# Interpolate at specific x and y points
x_values = np.array([1.0, 2.0, 3.0])
y_values = np.array([4.0, 5.0, 6.0])
interpolated_values = map_function(x_values, y_values)
```

## Examples

### Plotting a Characteristic Line

```python
import matplotlib.pyplot as plt

# Plot the curve
fig, ax = curve.to_figure()
plt.show()
```

### Plotting a Characteristic Map

```python
import matplotlib.pyplot as plt

# Plot the map
fig, ax = char_map.to_figure()
plt.show()
```

### Combining DCM Objects

You can combine DCM objects using set operations:

```python
# Union of two DCM objects
combined_dcm = dcm1 | dcm2

# Difference between two DCM objects
difference_dcm = dcm1 - dcm2

# Intersection of two DCM objects
intersection_dcm = dcm1 & dcm2

# Modifications between two DCM objects
modifications_dcm = dcm1 % dcm2
```

## Dependencies

- Python >= 3.11
- [NumPy](https://numpy.org/) >= 1.20.0
- [Pandas](https://pandas.pydata.org/) >= 1.5.0
- [Matplotlib](https://matplotlib.org/) >= 3.0.0
- [OpenPyXL](https://openpyxl.readthedocs.io/en/stable/) >= 3.1.0

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

Please ensure that your code adheres to the existing style and that all tests pass.

## Contact

Author: c0sogi  
Email: [dcas@naver.com](mailto:dcas@naver.com) or [cosogi1@gmail.com](mailto:cosogi1@gmail.com)

Feel free to reach out for questions or discussions.
