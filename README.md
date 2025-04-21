# interpretable_yield_prediction
Pytorch implementation of the method described in the paper [Interpretation of chemical reaction yields with graph neural additive network](#)

## Overview
- This study introduces an interpretable chemical reaction yield prediction method, which models the overall yield as a summation of component-wise contributions from each reaction participant. To enable interpretability, we propose a Graph Neural Additive Network (GNAN) architecture, which processes individual reaction components through shared neural networks and computes their contributions using a reaction-level context embedding. The final yield is predicted by aggregating these contributions. The model is trained with a tailored objective that emphasizes the impact of key components while reducing the influence of less significant ones.
- This GitHub repository provides running examples of the proposed method on the Buchwald-Hartwig and Suzuki-Miyaura datasets.

## Components
- **data/split/*** - original data files
- **data/*** - processed data files
- **data/get_data.py** - script for preprocessing the data files
- **model/*** - trained model files
- **result/*** - interpretation results
- **dataset.py** - data structure & functions
- **model.py** - model architecture & training/inference functions
- **util.py** - functions used across scripts.
- **run.py** - script for model training & performance evaluation
- **visualization.ipynb** - Jupyter Notebook example for visualization of interpretatations

## Usage Example

### Data processing
- Train/test splits for the Buchwald-Hartwig and Suzuki-Miyaura datasets are located in the `./data/split/` directory.
- To preprocess the data, run the following command:
```python
python ./data/get_data.py
```
- The processed data files are stored in the `./data/` directory.

### Training an interpretation model
- To train the model on the Buchwald-Hartwig dataset, run the following command (data_id: 1 (Buchwald-Hartwig), split_id: 0, train_size: 2767):
```python
python run.py -d 1 -s 0 -t 2767
```
- To train the model on the Suzuki-Miyaura dataset, run the following command (data_id: 2 (Suzuki-Miyaura), split_id: 0, train_size: 4032):
```python
python run.py -d 2 -s 0 -t 4032
```
- The trained models are stored in the `./model/` directory.
- The predictions and interpretations are stored in the `./result/` directory.

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**

## Citation
