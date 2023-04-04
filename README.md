# [Learning Causal Discovery](https://arxiv.org/abs/2209.05598)
Could a Neural Network Understand Microprocessor?

## Table Of Contents
-  [Introduction](#introduction)
-  [Requirements](#requirements)
-  [Codebase](#codebase)
-  [Usage](#usage)

## Introduction  
Causal discovery (CD) from time-varying data is important in neuroscience, 
medicine, and machine learning. Techniques for CD include randomized 
experiments which are generally unbiased but expensive. It also includes 
algorithms like regression, matching, and Granger causality, which are 
only correct under strong assumptions made by human designers. However, 
as we found in other areas of machine learning, humans are usually not 
quite right and human expertise is usually outperformed by data-driven 
approaches. Here we test if we can improve causal discovery in a 
data-driven way. We take a perturbable system with a large number of 
causal components (transistors), the MOS 6502 processor, acquire the 
causal ground truth, and learn the causal discovery procedure represented
as a neural network. We find that this procedure far outperforms 
human-designed causal discovery procedures, such as Mutual Information, 
LiNGAM, and Granger Causality both on MOS 6502 processor and the NetSim 
dataset which simulates functional magnetic resonance imaging (fMRI) 
results. We argue that the causality field should consider, where possible, 
a supervised approach, where CD procedures are learned from large datasets
with known causal relations instead of being designed by a human 
specialist. Our findings promise a new approach toward improving CD 
in neural and medical data and for the broader machine learning community.
## Requirements
Clone the repo:
```
mkdir learning_causal_discovery
git clone https://github.com/CharonWangg/LearningCausalDiscovery.git learning_causal_discovery
```
### For MOS 6502 Simulation (Modified from and inspired by [Sim2600](https://github.com/ericmjonas/Sim2600)):  
* Create a Python 2.7 env
* Setup:
    ```
    conda activate env_py2.7
    cd learning_causal_discovery/nmos_simulation && pip install -r requirements.txt
    cd learning_causal_discovery/nmos_simulation && pip install -e .
    ```
### For NMOS 6502 Inference
* Create a Python 3.9 env
* Install requirements:
    ```
    conda activate env_py3.9
    cd learning_causal_discovery/run/tools && pip install -r requirements.txt
    ```
  
### For Experiments on NetSim
* Download NetSim dataset
```
cd learning_causal_discovery && mkdir .cache
cd .cache
wget https://www.fmrib.ox.ac.uk/datasets/netsim/sims.tar.gz
tar -xzvf sims.tar.gz
```

## Codebase (after preparation)
```
├── README.md
├── work_dir  # work directory for checkpoints and corresponding configs
├── nmos_simulation  # MOS 6502 simulation
│   ├── build # build directory
│   ├── causal_simulation
│   │   ├── simulation.py # regular simulation/perturbation
│   │   ├── adjacency_matrix_generation.py # adjacency matrix generation
│   │   ├── record_transistor_state.py # helper functions
│   ├── sim2600 # core simulation code
├── run  # learning procedure related
│   ├── scripts  # scripts for tests and dataset generation
│   ├── official_configs  # network configs
│   ├── notebooks  # inference and figures
│   ├── trian.py  # entry point for training
│   ├── train_all.sh  # shell script for training
│   ├── tools  # deep learning tool
```
## Usage
### MOS 6502 Simulation
First, to acquire the regular state sequences:
```
conda activate env_py2.7
cd learning_causal_discovery
python nmos_simulation/causal_simulation/simulation.py
```
Then, acquire the perturbed state sequences:
```
python causal_simulation/simulation.py --action Adaptive
```
Then, generate the adjacency matrix from the data generated mentioned above
```
python causal_simulation/adjacency_matrix_generation.py
```
### Learning Procedure
First, to generate the train/val/test .csv files from the state sequences and adjacency matrix acquired from
MOS 6502 Simulation.
```
conda activate env_py3.9
cd learning_causal_discovery
```
```
python run/scripts/make_csv.py
```
Train all the networks for MOS 6502 and NetSim mentioned in the paper (LSTM, TCN, Transformer, SLDisco)
* LSTM (regular, noise w/ 0.1std, noise w/ 0.3std, noise w/ 0.5std)
```
python run/train.py --cfg run/official_configs/patch_lstm_128_50ep_cosine_adamw_1e-3lr_256bs_0.05wd.py
python run/train.py --cfg run/official_configs/patch_lstm_128_w0.1noise.py
python run/train.py --cfg run/official_configs/patch_lstm_128_w0.3noise.py
python run/train.py --cfg run/official_configs/patch_lstm_128_w0.5noise.py
```
* TCN (regular, noise w/ 0.1std, noise w/ 0.3std, noise w/ 0.5std)
```
python run/train.py --cfg run/official_configs/patch_tcn_128_50ep_cosine_adamw_1e-3lr_256bs_0.05wd.py
python run/train.py --cfg run/official_configs/patch_tcn_128_w0.1noise.py
python run/train.py --cfg run/official_configs/patch_tcn_128_w0.3noise.py
python run/train.py --cfg run/official_configs/patch_tcn_128_w0.5noise.py
```
* Transformer 
  * for MOS 6502 (regular, noise w/ 0.1std, noise w/ 0.3std, noise w/ 0.5std)
  ```
  python run/train.py --cfg run/official_configs/patch_transformer_128_50ep_cosine_adamw_1e-3lr_256bs_0.05wd.py
  python run/train.py --cfg run/official_configs/patch_transformer_128_w0.1noise.py
  python run/train.py --cfg run/official_configs/patch_transformer_128_w0.3noise.py
  python run/train.py --cfg run/official_configs/patch_transformer_128_w0.5noise.py
  ```
  * for NetSim (regular)
  ```
  python run/train.py --cfg run/official_configs/netsim_selection_partial.py
  ```
* SLDisco 
  * for NetSim (5 nodes, 10 nodes, 15 nodes, 50 nodes)
  ```
  python run/train.py --cfg run/official_configs/sldisco_node_5_n_samples_10000.py
  python run/train.py --cfg run/official_configs/sldisco_node_10_n_samples_10000.py
  python run/train.py --cfg run/official_configs/sldisco_node_15_n_samples_10000.py
  python run/train.py --cfg run/official_configs/sldisco_node_50_n_samples_10000.py
  ```
* Here we upload all configs mentioned above but only with seed 42.
All saved checkpoints and configs should be in `learning_discovery/work_dir`
### Inference and Figures
For the notebooks (`learning_discovery/run/notebooks`), set the notebook work directory to `learning_discovery`

---
<p align=center><b>Made with 💚 at <a href="https://kordinglab.com"><img alt="KordingLab" src="https://avatars.githubusercontent.com/u/7226053?s=200&v=4" height="23px" /></a></b></p>

