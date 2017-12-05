# How to do XGBOOST parameter tuning

Reference: [AARSHAY JAIN ](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

## Parameter List of XGboost

1. General Parameters: Overall Functioning
   * `booster`: Select the type of model **gbtree** and **gblinear**
   * `slient` : 0 or 1
   * `nthread` : number of thread

2. Booster Parameters: Individual booster (tree/regression) at each step
   * `eta` : learning rate
   * `min_child_weight` : minimum sum of weights of all observations required in a child.
   * `max_depth` : maximum depth of a tree
   * `max_leaf_nodes` : maximum number of terminal nodes or leaves in a tree
   * `gamma` :  Gamma specifies the minimum loss reduction required to make a split. A node is split only when the resulting split gives a positive reduction in the loss function.
   * `max_delta_step` : In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. **(Helpful in logistic regression when class is extremely imbalanced)**
   * `subsample` : Denotes the fraction of observations to be randomly samples for each tree
   * `lambda` : L2 regularization term on weights
   * `alpha` : L1 regularization term on weight
   * `scale_pos_weight` : A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence

3. Learning Task Parameters: optimization performed
   * `objective` : defines the loss function to be minimized **(binary:logistic, multi:softmax, multi:softprob)**
   * `eval_metric` : metric to be used for validation data
   * `seed` : random number seed

## General Method for parameter tuning

1. Fix learning rate and number of estimators
   First give some default values:
   * `max_depth` = 5 (3 - 10)
   * `min_child_weight` = 1 (Small for unbalanced dataset)
   * `gamma` = 0 (or 0.1 - 0.2)
   * `subsample` = 0.8
   * `colsample_bytree` = 0.8
   * `scale_pos_weight` = 1
   * `eta` : 0.1

   ```python
      #Choose all predictors except target & IDcols
      predictors = [x for x in train.columns if x not in [target, IDcol]]
      xgb1 = XGBClassifier(
       learning_rate =0.1,
       n_estimators=1000,
       max_depth=5,
       min_child_weight=1,
       gamma=0,
       subsample=0.8,
       colsample_bytree=0.8,
       objective= 'binary:logistic',
       nthread=4,
       scale_pos_weight=1,
       seed=27)
       modelfit(xgb1, train, predictors)
   ```

2. Tune `max_depth` and `min_child_weight`
   First tune those two because it effact the result the most.
   * Use heavy grid seawrch.

   ```python
       param_test1 = {
       'max_depth':range(3,10,2),
       'min_child_weight':range(1,6,2)
       }

       gsearch1 = GridSearchCV(estimator = XGBClassifier(
       learning_rate =0.1,
       n_estimators=140,
       max_depth=5,
       min_child_weight=1,
       gamma=0,
       subsample=0.8,
       colsample_bytree=0.8,
       objective= 'binary:logistic',
       nthread=4,
       scale_pos_weight=1,
       seed=27,
       param_grid = param_test1,
       scoring='roc_auc',
       n_jobs=4,
       iid=False,
       cv=5)

       gsearch1.fit(train[predictors],train[target])
       gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
   ```

    * Find a better value based on the value we get from grid search

   ```python
   param_test2b = {
     'min_child_weight':[6,8,10,12]
   }
   gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
   min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
   param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
   gsearch2b.fit(train[predictors],train[target])
   ```

3. Tune `Gamma`

   * Show best `Gamma`

   ```python
   param_test3 = {
     'gamma':[i/10.0 for i in range(0,5)]
   }
   gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
   min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
   param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
   gsearch3.fit(train[predictors],train[target])
   gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
   ```

4. Tune subsample and colsample_bytree

   * Stage 1
   ```python
    param_test4 = {
      'subsample':[i/10.0 for i in range(6,10)],
      'colsample_bytree':[i/10.0 for i in range(6,10)]
    }
    gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
    min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
    objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch4.fit(train[predictors],train[target])
    gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
   ```

   * Stage 2
   ```python
    param_test5 = {
       'subsample':[i/100.0 for i in range(75,90,5)],
       'colsample_bytree':[i/100.0 for i in range(75,90,5)]
    }
    gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
    min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
    objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch5.fit(train[predictors],train[target])
   ```

5. Tuning Regularization Parameters

   * tune `reg_alpha` or `reg_lambda`
   ```python
   param_test6 = {
      'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
   }
   gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
   min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
   param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
   gsearch6.fit(train[predictors],train[target])
   gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
   ```

   * Try the value closer to optimal
   ```python
   param_test7 = {
      'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
   }
   gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
   min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
   param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
   gsearch7.fit(train[predictors],train[target])
   gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
   ```

6. Reducing Learning Rate

   * lower the learning rate and add more trees (Use cv function)
   ```python
   xgb4 = XGBClassifier(
      learning_rate =0.01,
      n_estimators=5000,
      max_depth=4,
      min_child_weight=6,
      gamma=0,
      subsample=0.8,
      colsample_bytree=0.8,
      reg_alpha=0.005,
      objective= 'binary:logistic',
      nthread=4,
      scale_pos_weight=1,
      seed=27)
   modelfit(xgb4, train, predictors)
   ```
