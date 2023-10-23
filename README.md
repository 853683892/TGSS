# TGSS

## summary

Molecular property prediction is an important task in drug discovery, with help of self supervised learning method, the performance of molecular property prediction could be imporved by utilizing large scale unlabeled dataset. In this paper, we propose a triple generative self-supervised learning method for molecular property prediction, called TGSS. 
## Dependencies

- Python >= 3.8
- Pytorch >= 1.9.0

## Usage

#### Pre-training 

`pretain-TGSS.py`

#### Downstream prediction

`molecule_finetune.py`

## Dataset

| Dataset | Task | Task type | #Molecule | Splits | Metric |
| ------- | ---- | --------- | --------- | ------ | ------ |
| FreeSolv | 1 | Regression | 642 | Random | RMSE |
| ESOL | 1 | Regression | 1128 | Random | RMSE |
| Lipophilicity | 1 | Regression | 4200 | Random | RMSE |
| BACE | 1 | Classification | 1513 | Random | ROC-AUC |
| BBBP | 1 | Classification | 2039 | Random | ROC-AUC |
| HIV | 1 | Classification | 41127 | Random | ROC-AUC |
| TOx21 | 12 | Classification | 7831 | Random | ROC-AUC |
| SIDER | 27 | Classification | 1427 | Random | ROC-AUC |






  
