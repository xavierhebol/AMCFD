# Boundary Conditions Module - Conversion Summary

## Overview
Successfully converted `fortran/mod_bound.f90` to `taichi/bound.py` for the AM-CFD Taichi implementation.

## Files Created

### 1. `taichi/bound.py`
Main boundary conditions module with the following functions:

#### Taichi Kernels:
- `apply_boundary_conditions_u()` - u-velocity boundary conditions (ivar=1)
  - Marangoni stress at top surface
  - No-slip at solid boundaries
  
- `apply_boundary_conditions_v()` - v-velocity boundary conditions (ivar=2)
  - Marangoni stress at top surface
  - No-slip at solid boundaries
  
- `apply_boundary_conditions_w()` - w-velocity boundary conditions (ivar=3)
  - No-slip at solid boundaries
  
- `apply_boundary_conditions_p()` - Pressure correction boundary conditions (ivar=4)
  - Zero at solid boundaries
  
- `apply_boundary_conditions_enthalpy()` - Enthalpy boundary conditions (ivar=5)
  - Top surface: Laser heating + radiation + convection losses
  - Bottom surface: Convection + radiation losses
  - Side surfaces: Convection losses

#### Main Function:
- `bound_condition()` - Entry point that dispatches to appropriate BC based on ivar

### 2. `taichi/test_bound.py`
Comprehensive test suite including:
- `test_velocity_boundary_conditions()` - Tests u, v, w, and pp BCs
- `test_enthalpy_boundary_conditions()` - Tests thermal BCs with laser heating
- `test_marangoni_effect()` - Tests Marangoni stress with temperature gradients

## Key Features

### Marangoni Effect Implementation
The top surface boundary condition includes Marangoni stress:
```
u_top = u_below + (fracl * dgdt * dT/dx) / (vis * dz/dz)
v_top = v_below + (fracl * dgdt * dT/dy) / (vis * dz/dz)
```

Where:
- `dgdt` = surface tension temperature coefficient (dγ/dT)
- `dT/dx`, `dT/dy` = temperature gradients at surface
- `fracl` = liquid fraction (Marangoni only active in liquid)
- `vis` = viscosity

### Thermal Boundary Conditions
**Top Surface (k=nk):**
```
H_top = H_below + (Q_laser - Q_radiation - Q_convection) / (k/dz)
```

**Other Surfaces:**
```
H_boundary = H_interior - Q_convection / (k/dx)
```

Where:
- `Q_radiation = emiss * sigma * (T^4 - T_amb^4)`
- `Q_convection = h * (T - T_amb)`

## Input Arguments

The `bound_condition()` function requires:
- `ivar`: int (1-5) - Variable type index
- `state`: State - Flow field variables (from data_structures.py)
- `grid`: GridParams - Grid geometry (from data_structures.py)
- `mat_props`: MaterialProps - Material properties (from data_structures.py)
- `physics`: PhysicsParams - Physical parameters (from data_structures.py)
- `simu_params`: SimulationParams - Simulation parameters including index bounds (from data_structures.py)
- `laser`: LaserState (optional, required for ivar=5) - Laser heat input

## Return Value

Returns `ahtoploss: float` - Total heat loss at top surface [W]
- Only meaningful for ivar=5 (enthalpy)
- Returns 0.0 for velocity and pressure BCs

## Key Differences from Fortran

1. **Indexing**: Python uses 0-based indexing vs Fortran 1-based
   - Fortran `i=2,nim1` → Python `i in range(1, nim1)`
   - Fortran `k=nk` → Python `k=nk-1`

2. **Array interpolation factors (fracx, fracy)**:
   - Not stored in GridParams (yet)
   - Computed on-the-fly in boundary kernels:
     ```python
     fracx = (grid.x[i] - grid.xu[i]) / (grid.x[i] - grid.x[i-1])
     ```

3. **Harmonic mean for viscosity**:
   - Properly interpolated to velocity face locations
   - Uses same formula as Fortran for consistency

4. **Heat loss bug preserved**:
   - Original Fortran line 110 has `hlossradia` twice (likely bug)
   - Preserved for exact conversion, should be reviewed

## Testing

Run the test suite:
```bash
python taichi/test_bound.py
```

Tests verify:
- ✓ No-slip velocity boundaries (u=v=w=0 at walls)
- ✓ Zero pressure correction at boundaries
- ✓ Thermal boundary conditions apply heat fluxes
- ✓ Marangoni effect creates surface velocities
- ✓ Heat loss calculation is positive

## Integration with main.py

The function signature matches the stub in `main.py`:
```python
def bound_condition(ivar: int, state: State, grid: GridParams, 
                    mat_props: MaterialProps, physics: PhysicsParams,
                    simu_params: SimulationParams, laser: LaserState = None) -> float:
```

**Note**: The return type should be `float` for heat loss (not `DiscretCoeffs` as in the original stub).

## SimulationParams Updates

Added the following index bounds to `SimulationParams` dataclass:
- `istat`, `jstat`, `kstat` - Start indices (0 in Python)
- `iend`, `jend`, `kend` - End indices (ni, nj, nk)
- `istatp1`, `jstatp1`, `kstatp1` - Start + 1
- `iendm1`, `jendm1`, `kendm1` - End - 1  
- `nim1`, `njm1`, `nkm1` - Dimension - 1

These are automatically computed in `__post_init__()` from ni, nj, nk.

## Future Improvements

1. Store `fracx`, `fracy`, `fracz` in GridParams for efficiency
2. Fix the double `hlossradia` in top surface heat loss
3. Add htci, htcj, htck1, htckn as separate parameters (currently all use `hconv`)
4. Consider adding symmetry plane boundary conditions
5. Add periodic boundary condition support
