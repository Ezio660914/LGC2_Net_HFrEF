# LGC2-Net: Local-to-global hierarchical fusion network for heart failure detection using electrocardiograms and phonocardiograms

This is the official implementation of:

*LGC2-Net: Local-to-global hierarchical fusion network for heart failure detection using electrocardiograms and phonocardiograms*

## Model

**Architecture:** The architecture of the model is implemented in `LGC2_Net_HFrEF/source/models/LGC2_Net.py`.

**Pre-trained Models:** To use our pre-trained models, please download model checkpoints from [Zenodo](), and unzip them into `LGC2_Net_HFrEF/model_checkpoints`.

## In-house dataset

**Download:** The in-house dataset for HFrEF detection used in this study is released on [FigShare](). Please unzip the dataset into `LGC2_Net_HFrEF/dataset`.

**Preprocessing:** The preprocessing steps including resampling (at 2000 Hz), normalization (mean-std), and band-pass filtering (25~400 Hz, PCG only). Code is presented in `LGC2_Net_HFrEF/source/data/load_data.py`.

## Environment Setup

1. Please create a virtual environment with `python==3.7.10`
2. Install main packages `tensorflow==2.2.0`
3. Install other essential packages listed in the `requirements.txt`

## Run Validation

* To validate the pre-trained LGC2-Net on **in-house five-fold dataset**, run `validate_five_folds.py` in the folder `LGC2_Net_HFrEF/source/experiments`. This will load the model from `LGC2_Net_HFrEF/model_checkpoint/LGC2_Net/model_fold_*.hdf5`, and validate waveform data in `LGC2_Net_HFrEF/dataset/HFrEF_5_folds`.
* To validate the pre-trained LGC2-Net on **in-house community scenario test set**, run `validate_community_scenario_test_set.py` in the folder `LGC2_Net_HFrEF/source/experiments`. This will load the model from `LGC2_Net_HFrEF/model_checkpoint/LGC2_Net/model_fold_*.hdf5`, and validate waveform data in `LGC2_Net_HFrEF/dataset/HFrEF_community_scenario_test_set`.