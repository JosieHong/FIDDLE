# FIDDLE

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**F**ormula **ID**entification from tandem mass spectra by **D**eep **LE**arning

The source code for the training and evaluation of FIDDLE, as well as for the inference of FIDDLE using results from SIRIUS and BUDDY, is provided (see detailed commands in `./running_scripts/`). A PyPI package and a website-based service for FIDDLE will be available soon. 

## Set up

### Requirements

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/), if not already installed.

2. Create the environment with the necessary packages:

```bash
conda env create -f environment.yml
```

3. (optional) Install [BUDDY](https://github.com/Philipbear/msbuddy) and [SIRIUS](https://v6.docs.sirius-ms.io/) following the respective installation instructions provided in each tool's documentation. 

### Pre-trained Model Weights

To use the pre-trained models, download the weights from the [release page](https://github.com/JosieHong/FIDDLE/releases/tag/v1.0.0):

- **Orbitrap models**:
  - `fiddle_tcn_orbitrap.pt`: formula prediction model on Orbitrap spectra
  - `fiddle_fdr_orbitrap.pt`: confidence score prediction model on Orbitrap spectra
- **Q-TOF models**:
  - `fiddle_tcn_qtof.pt`: formula prediction model on Q-TOF spectra
  - `fiddle_fdr_qtof.pt`: confidence score prediction model on Q-TOF spectra

## Usage

The input format is `mgf`, where `title`, `precursor_mz`, `precursor_type`, `collision_energy` fields are required. Here, we sampled 21 spectra from the EMBL-MCF 2.0 dataset as an example.

```mgf
BEGIN IONS
TITLE=EMBL_MCF_2_0_HRMS_Library000531
PEPMASS=129.01941
CHARGE=1-
PRECURSOR_TYPE=[M-H]-
PRECURSOR_MZ=129.01941
COLLISION_ENERGY=50.0
SMILES=[H]OC(=O)C([H])=C(C(=O)O[H])C([H])([H])[H]
FORMULA=C5H6O4
THEORETICAL_PRECURSOR_MZ=129.018785
PPM=4.844255818912111
SIMULATED_PRECURSOR_MZ=129.02032113281717
41.2041 0.410228
55.7698 0.503672
56.8647 0.461943
85.0296 100.0
129.0196 8.036902
END IONS
```

**Run FIDDLE!**

```bash
python run_fiddle.py --test_data ./demo/input_msms.mgf \
                    --config_path ./config/fiddle_tcn_orbitrap.yml \
                    --resume_path ./check_point/fiddle_tcn_orbitrap.pt \
                    --fdr_resume_path ./check_point/fiddle_fdr_orbitrap.pt \
                    --result_path ./demo/output_fiddle.csv --device 0
```

If you'd like to integrate the results from SIRIUS and BUDDY, please organize the results in the format shown in `./demo/buddy_output.csv` and `./demo/sirius_output.csv`, and provide them to run FIDDLE:

```bash
python run_fiddle.py --test_data ./demo/input_msms.mgf \
                    --config_path ./config/fiddle_tcn_orbitrap.yml \
                    --resume_path ./check_point/fiddle_tcn_orbitrap.pt \
                    --fdr_resume_path ./check_point/fiddle_fdr_orbitrap.pt \
                    --buddy_path ./demo/output_buddy.csv \
                    --sirius_path ./demo/output_sirius.csv \
                    --result_path ./demo/output_fiddle_all.csv --device 0
```

## Citation

```
@article {Hong2024.11.25.625316,
	author = {Hong, Yuhui and Li, Sujun and Ye, Yuzhen and Tang, Haixu},
	title = {FIDDLE: a deep learning method for chemical formulas prediction from tandem mass spectra},
	elocation-id = {2024.11.25.625316},
	year = {2024},
	doi = {10.1101/2024.11.25.625316},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Molecular identification through tandem mass spectrometry is fundamental in metabolomics, with formula identification serving as the initial step in the process. However, current computation-based methods for formula identification face challenges, such as limited accuracy and extensive running times, with some methods unable to predict formulas for relatively large molecules. The limitations may impede high-throughput workflows and diminish overall research efficiency and success. To address these issues, we introduce FIDDLE (Formula IDentification by Deep LEarning using mass spectrometry), a novel deep learning-based method for formula identification. Our training and evaluation dataset comprises over 38,000 molecules and 1 million tandem mass spectra (MS/MS) acquired by using various Quadrupole time-of-flight (Q-TOF) and Orbitrap mass spectrometers. Comparative analyses demonstrate that FIDDLE accelerates formula identification by more than 10-fold and achieves top-1 and top-5 accuracies of 88.3\% and 93.6\%, respectively, surpassing state-of-the-art (SOTA) methods based on top-down (SIRIUS) and bottom-up (BUDDY) approaches by an average of over 10\%. On external benchmarking metabolomics datasets with novel compounds, FIDDLE significantly outperforms the SOTA methods, achieving on average the top-3 accuracy of 72.3\%. Furthermore, combining FIDDLE with existing methods such as BUDDY further improves performance, which achieves a higher top-3 accuracy of 79.0\%.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/11/28/2024.11.25.625316},
	eprint = {https://www.biorxiv.org/content/early/2024/11/28/2024.11.25.625316.full.pdf},
	journal = {bioRxiv}
}
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

