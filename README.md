# RNN-MCTS-with-In-Loop-xTB-Evaluation

De novo molecular generator for polyimide (PI) repeat-unit design based on the ChemTSv2
framework and semi-empirical quantum chemical calculations.

This program integrates xTB-based property evaluation into the molecular generation loop,
enabling reinforcement-learning-driven exploration of chemical space guided by
ionization potential (IP) and electron affinity (EA).
# Overview
This repository provides an extension of the ChemTSv2 framework for property-driven
molecular design. The original ChemTSv2 algorithm combines Monte Carlo Tree Search (MCTS)
with a recurrent neural network (RNN) to generate chemically valid molecules in SMILES format.

In this work, ChemTSv2 is extended by:
- Incorporating xTB-based quantum chemical calculations directly into the reward evaluation
  loop (in-loop evaluation),
- Designing a customized reward function targeting IP/EA optimization,
- Training a task-specific RNN model on polyimide-related molecular fragments.
