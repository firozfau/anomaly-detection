# MUTANT
For KBS-submission
> Robust Anomaly Detection for Multivariate Time Series through Temporal GCNs and Attention-based VAE

## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.21.2
* torch==1.9.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0
* tqdm==4.62.3
* git+https://github.com/thu-ml/zhusuan.git
* git+https://github.com/haowen-xu/tfsnippet.git@v0.2.0-alpha1
* pip install git+https://github.com/haowen-xu/tfsnippet.git
*pip install git+https://github.com/username/repository.git

## Datasets
### Link
The used datasets are available at:
* MSL&SMAP https://github.com/khundman/telemanom 
* https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl?resource=download


### Preprocess the data
`python data_preprocess.py <dataset name>`

exmpale: 

`Step-1:`
`python data_preprocess.py SMAP`

where `<dataset>` is one of `SMAP`, then you will get `<dataset>_train.pkl`, `<dataset>_test.pkl` and `<dataset>_test_label.pkl` in folder ‘processed’.

`Step-2:`
## Run Code
`python main.py`

If you want to change the default configuration, you can edit `ExpConfig` in `main.py`. For example, if you want to change dataset, you can change the value of 'dataset'.

`Step-3:`
After complete the traing then 
`dataSynchronizationStatus =False  in main.py`


### Recommended parameter settings
SMAP: `out_dim=5`, `window_length=20`, `hidden_size=100`, `latent_size=100`

## Main -Proposed paper link:
https://www.sciencedirect.com/science/article/abs/pii/S0950705123004756

## Main Source code based on Paper 
https://github.com/Coac-syf/MUTANT
