from functools import partial
from typing import Dict

from src.data.raw_data.helpsteer import HelpSteerRDP, HelpSteerRDPWithConflict

from .raw_data import (
    RawDatasetPreprocessor,
    PKUSafeRlhf10KRDP, PKUSafeRlhf10KRDPWithConflict,
)
from .raw_data.utils import DEFAULT_PROMPT_TEMPLATE


REAL_DATASET_CONFIGS: Dict[str, RawDatasetPreprocessor] = {
    ##### PKU-SafeRLHF (https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) #####
    **{
        f"PKU-Alignment/PKU-SafeRLHF-10K-{dimension}": partial(PKUSafeRlhf10KRDP, dimension=dimension)
        for dimension in ["safer", "better"]
    },
    **{
        f"PKU-Alignment/PKU-SafeRLHF-10K-{dimension}-{fraction}": partial(PKUSafeRlhf10KRDPWithConflict, dimension=dimension, conflicting_fraction=fraction)
        for dimension in ["safer", "better"]
        for fraction in [0.0, 0.3, 0.5, 0.6, 0.9]
    },

    ##### HelpSteer (https://huggingface.co/datasets/nvidia/HelpSteer) #####
    "nvidia/HelpSteer": HelpSteerRDP,
    **{
        f"nvidia/HelpSteer-pairwise-{dimension}": partial(HelpSteerRDP, dimension=dimension)
        for dimension in ["overall", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    },
    **{
        f"nvidia/HelpSteer-pairwise-{dimension}-{fraction}": partial(HelpSteerRDPWithConflict, dimension=dimension, conflicting_fraction=fraction)
        for dimension in ["correctness", "verbosity", "helpfulness"]
        for fraction in [0.0, 0.2, 0.3, 0.6, 0.9]
    }
}


# !WARNING: Synthetic datasets are WIP. These configs are just placeholders 
SYNTHETIC_DATASET_CONFIGS = {

}


# MERGE two dicts
DATASET_CONFIGS = {**REAL_DATASET_CONFIGS, **SYNTHETIC_DATASET_CONFIGS}
