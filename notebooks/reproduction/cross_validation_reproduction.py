# Install ChemProp and download CheMeleon weights before running this script
# from urllib.request import urlretrieve
# urlretrieve(r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt", "chemeleon_mp.pt")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from torch.utils.data import Subset

from src.model import GNN_net
from src.trainer import MultitaskTrainer
from data.datasets import OnePropTRAINDataset

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from tqdm.notebook import tqdm

from lightning import pytorch as pl
from chemprop import data, models
from chemprop import nn as cpnn
from chemprop.models.utils import save_model, load_model
from chemprop.models.model import MPNN
from chemprop.nn.metrics import MAE

import warnings
warnings.simplefilter(action='ignore')

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


ml_list = [RandomForestRegressor(),
           LGBMRegressor(verbose = -1),
           XGBRegressor(verbose = False),
           CatBoostRegressor(verbose = False)
           ]

ml_name_list = ['RF',
                'LGBM',
                'XGB',
                'CatBoost']

hyperparams = {'KSOL': {'epochs': 100, 'lr': 1e-4},
               'LogD': {'epochs': 100, 'lr': 1e-3},
               'HLM': {'epochs': 100, 'lr': 1e-4},
               'MLM': {'epochs': 100, 'lr': 1e-3},
               'MDR1-MDCKII': {'epochs': 100, 'lr': 1e-4}
               }

name_map = {'KSOL': 'y_sol', 'HLM': 'y_clint', 'MLM': 'y_clint', 'LogD': 'y_logd', 'MDR1-MDCKII': 'y_clint'}

chemeleon_model = torch.load("chemeleon_mp.pt", weights_only=True)
chemeleon_mp = cpnn.BondMessagePassing(**chemeleon_model['hyper_parameters'])
chemeleon_mp.load_state_dict(chemeleon_model['state_dict'])

def run_cross_val(task, k_fold = 5, replicate = 2):
    task_dataset = OnePropTRAINDataset(root = 'pytorch_data', dataset = f'{task}_train')

    # for the ML models
    df = pd.read_csv(f'csv_files/{task}_train.csv')
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize = 2048)
    mols = [Chem.MolFromSmiles(smi) for smi in df['std_smiles']]
    fp_list = [mfpgen.GetFingerprintAsNumPy(mol) if mol else None for mol in mols]
    X = np.array(fp_list)
    y = df['y_value'].to_numpy()

    # For ChemProp
    smis = df.loc[:, 'std_smiles'].values
    ys = df.loc[:, 'y_value'].values
    datapoints = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    all_indices = np.arange(len(task_dataset))
    full_mae_results = []
    full_pearson_results = []

    for rep in range(replicate):
        kf = KFold(n_splits = k_fold, shuffle=True)
        folds = list(kf.split(all_indices))

        for fold, (train_index, val_index) in enumerate(folds):
            print(f"Starting fold {fold + 1}/{k_fold}")
            df_val = df.iloc[val_index]

            mae_results = {}
            pearson_results = {}

            # GIN model
            train_data = Subset(task_dataset, train_index)
            val_data = Subset(task_dataset, val_index)

            model = GNN_net(num_gnn_layers = 3, graph_pooling = 'attention', JK = 'last',
                            h_dim = 256, ffn_dim = 64
                            )
            model.load_state_dict(torch.load('models/leaderboard_models/best_pretrained_model_full_data.pt'))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trainer = MultitaskTrainer(model, tasks = ['y_sol', 'y_logd', 'y_clint'], device = device, mode = 'finetune')
            gin_pred = trainer.cross_val(task_name = name_map[task],
                                        train_dataset = train_data,
                                        test_dataset = val_data,
                                        epochs = hyperparams[task]['epochs'],
                                        lr = hyperparams[task]['lr']
                                        )
            df_val['GIN_pred'] = gin_pred.numpy()
            gin_mae = mean_absolute_error(df_val['y_value'], df_val['GIN_pred'])
            gin_pearson = pearsonr(df_val['y_value'], df_val['GIN_pred'])[0]

            mae_results['GIN_MAE'] = gin_mae
            pearson_results['GIN_PearsonR'] = gin_pearson
            print(f'   GIN MAE: {gin_mae}')

            # ChemProp model
            chemprop_train_data = data.MoleculeDataset([datapoints[idx] for idx in train_index])
            chemprop_val_data = data.MoleculeDataset([datapoints[idx] for idx in val_index])
            train_loader = data.build_dataloader(chemprop_train_data, batch_size = 64, num_workers = 2)
            val_loader = data.build_dataloader(chemprop_val_data, shuffle = False, batch_size = 32, num_workers = 2)

            ffn = cpnn.RegressionFFN(criterion = MAE())
            chemprop_model = models.MPNN(cpnn.BondMessagePassing(), cpnn.AttentiveAggregation(output_size = 300), ffn)
            trainer = pl.Trainer(max_epochs = 50, logger = False, enable_checkpointing = False, enable_progress_bar = False)      # 100 or 50
            trainer.fit(chemprop_model, train_loader)
            with torch.inference_mode():
                trainer = pl.Trainer(logger = False, enable_checkpointing = False)
                test_preds = trainer.predict(chemprop_model, val_loader)
            test_preds = np.concatenate(test_preds, axis=0)
            df_val['ChemProp_pred'] = test_preds
            chemprop_mae = mean_absolute_error(df_val['y_value'], df_val['ChemProp_pred'])
            chemprop_pearson = pearsonr(df_val['y_value'], df_val['ChemProp_pred'])[0]

            mae_results['ChemProp_MAE'] = chemprop_mae
            pearson_results['ChemProp_PearsonR'] = chemprop_pearson
            print(f'   ChemProp MAE: {chemprop_mae}')

            # CheMeleon model, takes very long time
            ffn = cpnn.RegressionFFN(input_dim = chemeleon_mp.output_dim, criterion = MAE())
            chemeleon = models.MPNN(chemeleon_mp, cpnn.MeanAggregation(), ffn, batch_norm=False)      # CheMeleon must use MeanAggregation
            trainer = pl.Trainer(max_epochs = 100, logger = False, enable_checkpointing = False, enable_progress_bar = False)
            trainer.fit(chemeleon, train_loader)
            with torch.inference_mode():
                test_preds = trainer.predict(chemeleon, val_loader)
            test_preds = np.concatenate(test_preds, axis=0)
            df_val['CheMeleon_pred'] = test_preds
            chemprop_mae = mean_absolute_error(df_val['y_value'], df_val['CheMeleon_pred'])
            chemprop_pearson = pearsonr(df_val['y_value'], df_val['CheMeleon_pred'])[0]

            mae_results['CheMeleon_MAE'] = chemprop_mae
            pearson_results['CheMeleon_PearsonR'] = chemprop_pearson
            print(f'   CheMeleon MAE: {chemprop_mae}')

            # ML models
            X_train = X[train_index]
            X_val = X[val_index]
            y_train = y[train_index]
            y_val = y[val_index]

            for ml, ml_name in zip(ml_list, ml_name_list):
                ml.fit(X_train, y_train)
                y_pred = ml.predict(X_val)
                df_val[f'{ml_name}_pred'] = y_pred
                method_mae = mean_absolute_error(y_val, y_pred)
                method_pearson = pearsonr(y_val, y_pred)[0]

                mae_results[f'{ml_name}_MAE'] = method_mae
                pearson_results[f'{ml_name}_PearsonR'] = method_pearson
                print(f'   {ml_name} MAE: {method_mae}')

            full_mae_results.append(mae_results)
            full_pearson_results.append(pearson_results)

            df_val.to_csv(f'cross_val_preds/{task}_rep_{rep}_fold_{fold}.csv', index = False)

    full_mae_df = pd.DataFrame(full_mae_results)
    full_mae_df.to_csv(f'cross_val_results/{task}_MAE.csv', index = False)
    full_pearson_df = pd.DataFrame(full_pearson_results)
    full_pearson_df.to_csv(f'cross_val_results/{task}_PearsonR.csv', index = False)

    return full_mae_df, full_pearson_df

if __name__ == "__main__":
    os.mkdir('cross_val_preds')
    os.mkdir('cross_val_results')
    for task in ['KSOL', 'LogD', 'HLM', 'MLM', 'MDR1-MDCKII']:
        full_mae_df, full_pearson_df = run_cross_val(task, k_fold = 5, replicate = 5)
        sns.boxplot(full_mae_df)
        plt.show()