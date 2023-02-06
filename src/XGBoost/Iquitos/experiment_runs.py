from experiment_setup import run_experiment
import numpy as np

param_grid = {'window': 5,
              'shift': -3,
              'drop_threshold': 0.05,
              'train_amount': 0.6,
              'booster': 'gbtree',
              'n_estimators': 20,
              'max_depth': 4,
              'learning_rate': 1,
              'contamination': 0.15}

train_amounts = np.linspace(0.5, 0.95, 10)

for train_amount in train_amounts:
    param_grid['train_amount'] = train_amount

    run_experiment(**param_grid)
