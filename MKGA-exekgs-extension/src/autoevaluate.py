from omegaconf import DictConfig
import hydra
import logging
import os
import numpy as np
import pandas as pd
import torch

from main import evaluate_approach

log = logging.getLogger(__name__)

def _calc_df()-> pd.DataFrame:
    """
    Creates df object based on results wihtin "../data/predicted"
    Used to skip allready predicted configs in multi embedding framework

    Returns:
        pd.DataFrame: dataframe with prediction results.
    """  
    if len(os.listdir("data/predicted/"))==0:
        return pd.DataFrame(columns=["count","dataset","eval_method","augment","embedder"])
    df = pd.DataFrame(columns=["full_name"])  
    dfs = []
    for entry in os.listdir('data/predicted/'):

        dfs.append(pd.DataFrame(
            [[
                entry,
            ]],
            columns=["full_name"]
        ))

    df = pd.concat(dfs,ignore_index=True)
    df['base_name']= df['full_name'].str.extract(r'(.*)\$[0-9]+\.csv')
    df['count'] = df.groupby('base_name')["base_name"].transform("count")
    df = df.groupby('base_name').mean().round(3)
    df = df.reset_index()
    df['dataset']= df['base_name'].str.extract(r'(.*?)\+.*')
    df['eval_method']= df['base_name'].str.extract(r'.*\$([A-Z]+)')
    df['augment'] = df['base_name'].str.extract(r'\+(.*?)\$.*')
    df['embedder'] = df['base_name'].str.extract(r'\$(.*?)\$')
    df = df.drop(columns='base_name')

    return df

@hydra.main(version_base=None, config_path="../config", config_name="multiple")
def multiple(cfg: DictConfig) -> None:
    """
    Evaluates multiple GDA approaches with multiple datasets, embedders and classifiers.
    Evaluation ist based on hydra settings in "../config/multiple.yaml"
    
    Args:
        cfg (DictConfig): hydra config file
    """  
    torch.cuda.empty_cache() # prevent memory overflow

    log.info("Load auto evaluater...")
    # loadd result dataframe and configs
    df = _calc_df()
  
    for dataload in cfg['schedule']['dataload']:
        for augment in cfg['schedule']['augment']:
            for embed in cfg['schedule']['embed']:
                cfg['pipeline']['dataload'] = dataload
                cfg['pipeline']['augment'] = augment
                cfg['pipeline']['embed'] = embed
                iterations = cfg['schedule']['iterations']
                if len(df[(df["dataset"] == dataload)
                   & (df['embedder'] == embed) & (df['augment']== augment)])> 0:
                    print(df[(df["dataset"] == dataload)
                    & (df['embedder'] == embed) & (df['augment']== augment)]['count'].values[0])
                    iterations = iterations- df[(df["dataset"] == dataload)
                    & (df['embedder'] == embed) & (df['augment']== augment)]['count'].values[0]

                if iterations > 0:
                    for i in range(iterations):
                        log.info(f"processing {dataload} | {augment} | {embed} (Nr. {i}/{iterations})")             
                        evaluate_approach(cfg)
                else:
                    log.info(f"skipping {dataload} | {augment} | {embed} -> iterations reached.")



if __name__ == '__main__':
    multiple()
