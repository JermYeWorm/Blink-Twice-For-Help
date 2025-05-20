# Digital Design Automation Scripts

This directory contains scripts for automating synthesis, simulation, and place & route tasks using Cadence tools. The top module in this project should be `snn_fixed_16_2_4_4_2`. Substitute all instances of `my_top_module` in the examples with this module name.

## Folder Structure

- **lib/**  
  Contains standard cell libraries and any custom library files required for synthesis, simulation, or PnR.

- **PnR/**  
  Place and Route (PnR) directory for Cadence Innovus.
  - `run_PnR`: Main script to launch PnR.
  - **scripts/**: TCL scripts for floorplanning, placement, routing, and verification.
  - **work/**: Working directory for PnR outputs and intermediate files.
    - `design/`: Design database and intermediate files generated during PnR.
    - `log/`: Log files from PnR tool runs.
    - `output/`: Final output files from PnR, such as GDSII, DEF, and other deliverables.
    - `report/`: Reports generated during PnR, including timing, area, and DRC/LVS results.
    - `timing/`: Detailed timing analysis reports.
    - `work/`: Additional temporary or intermediate files used by the PnR process.

- **simulation/**  
  Simulation environment for Cadence Xcelium.
  - `run_sim`: Main script to launch simulation.
  - **script/**: Helper scripts for simulation setup.
  - **sdf/**: Contains SDF files and command scripts for timing simulation.
  - **testbench/**: Testbench source files for simulation.
  - **work/**: Working directory for simulation outputs and intermediate files.
    - Stores compiled simulation libraries, simulation logs, waveform files, and any temporary files generated during simulation runs.

- **source/**  
  RTL source files (Verilog) for the project.
  - `fixed_16_synapse.v`, `IZ_pipeline_16.v`, `snn_fixed_16_2_4_4_2.v`: Main design files.

- **synthesis/**  
  Synthesis environment for Cadence Genus.
  - `run_syn`: Main script to launch synthesis.
  - **scripts/**: Helper scripts for synthesis setup.
  - **work/**: Working directory for synthesis outputs, logs, and reports.
    - `log/`: Log files from synthesis runs.
    - `report/`: Reports generated during synthesis, such as timing, area, and utilization.
    - `output/`: Final output files from synthesis, such as the final netlist and summary reports.

---

## Scripts Overview

### 1. `run_syn`
This script is used for running synthesis using Cadence Genus.

#### Features:
- Supports GUI and non-GUI modes.
- Allows specifying the top module for synthesis.

#### Usage:
```bash
./run_syn [options]
```

#### Options:
- `-gui`: Launches the synthesis process in GUI mode.
- `-top <module_name>`: Specifies the top module for synthesis.

#### Example:
```bash
./run_syn -gui -top my_top_module
```

---

### 2. `run_sim`
This script is used for running simulations using Cadence Xcelium.

#### Features:
- Supports cleaning the work directory.
- Allows compiling source files and standard cells.
- Enables simulation with or without GUI.
- Supports waveform viewing.

#### Usage:
```bash
./run_sim [options]
```

#### Options:
- `-clean`: Cleans the work directory.
- `-compile`: Compiles the source files.
- `-sim`: Runs the simulation (also triggers compilation).
- `-cells`: Includes standard cells during compilation.
- `-gui`: Runs the simulation in GUI mode.
- `-waveform`: Opens the waveform viewer.
- `-top <module_name>`: Specifies the top module for simulation.

#### Example:
```bash
./run_sim -sim -gui -top my_top_module
```

---

### 3. `run_PnR`
This script is used for running place and route using Cadence Innovus.

#### Features:
- Supports GUI and non-GUI modes.
- Allows specifying the top module for place and route.

#### Usage:
```bash
./run_PnR [options]
```

#### Options:
- `-gui`: Launches the place and route process in GUI mode.
- `-top <module_name>`: Specifies the top module for place and route.

#### Example:
```bash
./run_PnR -gui -top my_top_module
```

---

## Notes
- Ensure that the required Cadence tools (Genus, Xcelium, and Innovus) are installed and properly configured in your environment.
- The scripts assume specific directory structures for source files, testbenches, and synthesis outputs. Adjust paths in the scripts if necessary.