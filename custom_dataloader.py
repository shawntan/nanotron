
from typing import Dict, Generator, Iterator, List, Optional, Union
from nanotron.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
    _get_dataset_mix
)
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
    DownloadConfig
)

def get_datasets(
    hf_dataset_or_datasets: Union[dict, str],
    hf_dataset_config_name: str,
    splits: Optional[Union[List[str], str]] = ["train", "test"],
) -> "DatasetDict":
    """
    Function to load dataset directly from DataArguments.

    Args:
        hf_dataset_or_datasets (Union[dict, str]): dict or string. When all probabilities are 1, we concatenate the datasets instead of sampling from them.
        splits (Optional[List[str]], optional): Section of the dataset to load, defaults to "train", "test"
            Can be one of `train_ift`, `test_rl`, or `..._rm` etc. H4 datasets are divided into 6 subsets for training / testing.

    Returns
        DatasetDict: DatasetDict object containing the dataset of the appropriate section with test + train parts.
    """

    if isinstance(splits, str):
        splits = [splits]

    if isinstance(hf_dataset_or_datasets, dict):
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        raw_datasets = _get_dataset_mix(hf_dataset_or_datasets, splits=splits)
    elif isinstance(hf_dataset_or_datasets, str):
        # e.g. Dataset = "HuggingFaceH4/testing_alpaca_small"
        # Note this returns things other than just train/test, which may not be intended
        raw_datasets = DatasetDict()
        for split in splits:
            download_config = DownloadConfig(resume_download=True, num_proc=32, max_retries=10)
            raw_datasets[split] = load_dataset(
                hf_dataset_or_datasets,
                hf_dataset_config_name,
                split=split,
                download_config=download_config
            )
    else:
        raise ValueError(f"hf_dataset_or_datasets must be a dict or string but is {type(hf_dataset_or_datasets)}")

    return raw_datasets
