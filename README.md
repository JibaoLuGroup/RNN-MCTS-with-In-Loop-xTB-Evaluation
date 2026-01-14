# RNN-MCTS-with-In-Loop-xTB-Evaluation

De novo molecular generator for polyimide (PI) repeat-unit design based on the ChemTSv2
framework and semi-empirical quantum chemical calculations.

This program integrates xTB-based property evaluation into the molecular generation loop,
enabling reinforcement-learning-driven exploration of chemical space guided by
ionization potential (IP) and electron affinity (EA).
# Overview
This repository provides an extension of the ChemTSv2 framework(https://github.com/molecule-generator-collection/ChemTSv2) for property-driven
molecular design. The original ChemTSv2 algorithm combines Monte Carlo Tree Search (MCTS)
with a recurrent neural network (RNN) to generate chemically valid molecules in SMILES format.

In this work, ChemTSv2 is extended by:
- Incorporating xTB-based quantum chemical calculations directly into the reward evaluation
  loop (in-loop evaluation),
- Designing a customized reward function targeting IP/EA optimization,
- Training a task-specific RNN model on polyimide-related molecular data set.
# Requirements
- Python >= 3.8
- python: 3.11
- rdkit: 2023.9.1
- tensorflow: 2.14.1
- pyyaml
- pandas: 2.1.3
- joblib
- ChemTSv2
- xTB (GFN2-xTB, IPEA-xTB)
- NumPy
- PyYAML
# Usage
The main molecular generation workflow is executed through the ChemTSv2 interface.

1. Install ChemTSv2 following the official instructions.
2. Clone this repository and copy the provided files into the corresponding directories of your ChemTSv2 installation (e.g., reward module into ChemTSv2 reward directory and configuration YAML into ChemTSv2 config directory).
3. Run ChemTSv2 with the selected configuration file.

# License
This package is distributed under the MIT License.
