# How to deal with Overfitting in Kaggle

by Zhuoran Wu <zw118@georgetown.edu>

## 1. Do not depend on `Train Result`

Just ignore them.
* Train Error.
* Train AUC.

## 2. Feature Select depended on `Cross Validation`

Use `Cross Validation` to select feature, instead of other approaches such as `T-Tests` and `Lasso`.

## 3. Make sure `Cross Validation` that is consistently "directionally correct"

Improvements in your cv to be reflected in equal Improvements on the lb.

## 4. Weight-Decay (Regularization)

* L1 Regularization
* L2 Regularization

Add weight to Cose function directly.

## 5. Add Noise
* Add noise in the input.
* Add noise in the weight.
* Change the output of single cell from `binary` to `random`.

## 6. Model Ensemble
* Bagging
* Boosting
* Dropout: Leave out randomly some node in the hidden layer.

## 7. Bayes Method
* Still unclear for me.

## 8. Early Stop

## 9. Batch Normalization
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v1.pdf)
