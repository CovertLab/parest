# parest

Kinetic model parameter estimation workflow, currently implemented for modeling *E. coli*'s glycolytic pathway.

## Dependencies
- Python 2.7.5 or equivalent
- NumPy, SciPy, matplotlib
See `requirements.txt` for specific version information.

## Usage

### Basic parameter estimation
To run a single, standard optimization, call `python optimize.py`.  The output parameter values will be saved as `optimized_pars.npy`.

### Specific parameter estimation
To run a specific optimization problem, call `python main.py` with the appropriate options (see `python main.py -h` for arguments).  For example, to run the optimization using the smallest saturation penalty (see text) and see #32,

```bash
python main.py --seed=32 --problem=all_scaled_upper_sat_limits_1e-1
```

In this case the results (parameters and objective term values) will be saved to `out/all_scaled_upper_sat_limits_1e-1\seed-32\`.

You can also use `main_alt.py` to run optimizations, with the added functionality of controlling the output directory and enabling the 'naive' perturbation approach.

### Gathering output

Output can be collected by calling `python gather_output.py <output directory>`.  It will throw an exception if data is missing.  At the moment, this script assumes that the results for seeds 0-299 are present.

### Validation

Single sets of output parameters can be validated by calling `bash python validate_single.py`.  Gathered sets of parameter values can be validated by `python validate_model.py <output directory>`.  These files should give consistent results although `validate_single.py` may fall out of date (it exists as a weakly supported convenience).

### Simulation

Model ODEs and related equations are defined in `equations.py`; for usage, see `validate_model.py`.

## Extensions

The parameter estimation system is meant to be generic however extensions may require modifying code.

### Adding training data

Model data (parameter values and network structure) are stored in the `data` subdirectory.  New parameter value data can be added here, and will automatically be incorporated into the optimization problem.

### Adding optimization problems

Problem definitions are found in `problems.py`.

### Changing the network structure

The network structure is also defined in the `data` subdirectory.  New reactions will need to be added to the constant `ACTIVE_REACTIONS` in `structure.py`.  If a new target flux constraint (or similar) is desired, it will need to be defined in `optimize.py`.

### Changing the kinetic rate laws

Kinetic rate laws are automatically assembled using the procedure described in the text.  There is currently no option to use other kinetic rate law schemes.

## Manuscript data and figures

### Output data

The data used to generate all output is committed in the directory structure.  It has already been gathered up and validated for its dynamic viability.  If you wish to regenerate the data, you will need to call `main.py` or `main_alt.py` with the correct arguments for 300 seeds, then re-gather and output the data.  System independence of results should be true but is difficult to verify.

### Figures

Figure generation files are found under the `figures` subdirectory.  They should be executed from the root `parest` directory.

### Tables

Table generation files are found under the `tables` subdirectory.  They should be executed from the root `parest` directory.
