# interpretable_yield_prediction
Pytorch implementation of the method described in the paper [Interpretation of chemical reaction yields with graph neural additive network](#)

## Components
- **data/*** - data files used
- **data/get_data.py** - script for preprocessing the data files
- **model/*** - trained model files
- **result/*** - interpretation results
- **dataset.py** - data structure & functions
- **model.py** - model architecture & training/inference functions
- **util.py**
- **run.py** - script for model training & performance evaluation
- **visualization.ipynb** - visualization example of interpretatations

## Usage Example
`python run.py -d 1 -s 0 -t 2767`
`python run.py -d 2 -s 0 -t 4032`

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**

## Citation
