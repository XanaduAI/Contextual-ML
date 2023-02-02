# Contextuality and inductive bias in quatum machine learning models: numerical study

This repository contains the code (numerics.ipynb) used to generate the plots of section 8 'outperforming classical
surrogates' in arXiv:XXXXX. 

The quantum circuit simulation is done in pennylane. Both the quantum and surrogate training use JAX for vectorisation
and gradient computation.

Dependencies:
- pennylane 
- JAX
- optax

The files `X_data.npy` and `Y_data.npy` contain the input data and labels in numpy array format. 

please email any questions to bowles.physics@gmail.com.
