# Storm Surge ML Emulators for the Northern Adriatic Sea

This repository contains the implementation of machine learning (ML) emulators for storm surge prediction and statistical downscaling in the northern Adriatic Sea, developed within the framework of the study Campos-Caba et al. (2025). 

The project benchmarks a range of ML architectures, from simple linear approaches to deep recurrent neural networks, against a high-resolution hydrodynamic model (SHYFEM-MPI) optimized for the representation of extreme storm surge events in the northern Adriatic Sea. 

---

# Objectives

The main goals of this repository are:

* Develop computationally efficient ML emulators for storm surge prediction
* Compare different ML architectures for extreme-event representation
* Evaluate the impact of alternative loss functions on surge extremes
* Reproduce high-resolution coastal surge dynamics using basin-scale predictors
* Provide reproducible training and operational inference workflows

---

# Implemented ML Emulators

The repository includes implementations of:

* Multivariate Linear Regression (MLR)
* Multilayer Perceptron (MLP)
* Recurrent Neural Networks (RNN)
* Hybrid Recurrent Neural Networks (RNNh)
* Long Short-Term Memory networks (LSTM)
* Hybrid Long Short-Term Memory networks (LSTMh)

Both standard MSE-based training and the custom MADc² loss function are supported. 

---

# MADc² Loss Function

A central contribution of this work is the implementation of the **MADc²** loss function, specifically designed to improve the representation of extreme storm surge events.

MADc² combines:

* Mean Absolute Deviation (MAD)
* Percentile-based Mean Absolute Deviation (MADp)

allowing the models to jointly optimize:

* pointwise prediction accuracy,
* and the statistical distribution of extremes. 

For further details on the MADc metric the reader is refered to Campos-Caba et al. (2024).

---

# Predictors

The emulators use basin-scale atmospheric and oceanographic predictors including:

* CMEMS sea surface height principal components
* ERA5 mean sea level pressure principal components
* ERA5 wind stress principal components
* Tides from FES2014

Principal Component Analysis (PCA) is applied independently to each predictor field, retaining the first seven principal components for each variable. 

---

# Study Area

The framework was developed and tested for:

* Punta della Salute (Venice Lagoon)
* Trieste

in the northern Adriatic Sea. 

---

# Main Features

* PyTorch-based implementation
* GPU acceleration support
* TBPTT (Truncated Backpropagation Through Time) training for recurrent models
* Operational inference workflows
* Predictor alignment utilities
* Robust inference preprocessing
* Multi-run reproducibility workflows
* Checkpoint and preprocessing export

---

# Operational Inference

The repository includes operational inference workflows for:

* unseen years,
* external predictor datasets,
* and out-of-sample storm surge events.

Inference pipelines support:

* recurrent hidden-state propagation,
* predictor scaling consistency,
* robust input clipping,
* and post-processing low-pass filtering.

---

# Computational Efficiency

The implemented ML emulators achieve orders-of-magnitude reductions in computational cost relative to high-resolution hydrodynamic simulations while retaining strong skill in the representation of extreme storm surge events. 

---

# Requirements

Main dependencies:

* Python 3.x
* PyTorch
* NumPy
* Pandas
* SciPy
* scikit-learn
* Matplotlib
* joblib

---

# Notes on Temporal Alignment

Application on different locations could require station-specific temporal offsets to correctly align predictors and predictands due to differences in timestamp conventions between observational and reanalysis datasets.

These offsets are configurable through the YAML configuration files.

---

# Reference

Campos-Caba, R., Camus, P., Mazzino, A., Vousdoukas, M., Tondello, M., Federico, I., Causio, S., & Mentaschi, L. (2025). *Storm surge dynamics in the northern Adriatic Sea: comparing AI emulators with high-resolution numerical simulations. EGUsphere [preprint]. https://doi.org/10.5194/egusphere-2025-5313*.

Campos-Caba, R., Alessandri, J., Camus, P., Mazzino, A., Ferrari, F., Federico, I., Vousdoukas, M., Tondello, M., & Mentaschi, L. (2024). *Assessing storm surge model performance: what error indicators can measure the model's skill? Ocean Science, 20, 1513–1526. https://doi.org/10.5194/os-20-1513-2024*

---

