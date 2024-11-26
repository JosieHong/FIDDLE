# FIDDLE

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**F**ormula **ID**entification from tandem mass spectra by **D**eep **LE**arning

Here is the source code for all the experiments, including the training of FIDDLE, as well as the evaluation and testing of FIDDLE, BUDDY, and SIRIUS. A PyPI package and a website-based service for FIDDLE will be available soon.

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

The input format is `mgf`, where `title`, `precursor_mz`, `precursor_type`, `collision_energy` fields are required. Here, we sampled 100 spectra from the EMBL-MCF 2.0 dataset as an example.

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
                    --result_path ./demo/fiddle_output.csv --device 0
```

If you'd like to integrate the results from SIRIUS and BUDDY, please organize the results in the format shown in `./demo/buddy_output.csv` and `./demo/sirius_output.csv`, and provide them to run FIDDLE:

```bash
python run_fiddle.py --test_data ./demo/input_msms.mgf \
                    --config_path ./config/fiddle_tcn_orbitrap.yml \
                    --resume_path ./check_point/fiddle_tcn_orbitrap.pt \
                    --fdr_resume_path ./check_point/fiddle_fdr_orbitrap.pt \
                    --buddy_path ./demo/buddy_output.csv \
                    --sirius_path ./demo/sirius_output.csv \
                    --result_path ./demo/all_output.csv --device 0
```

## Testing on external public datasets

1. Download CASMI 2016 and EMBL from the [[MassBank of North America website]](https://mona.fiehnlab.ucdavis.edu/downloads) and CASMI 2017 from the [[CASMI website]](http://www.casmi-contest.org/2017/index.shtml). The data directory is structured as follows:

```bash
|- data
  |- casmi
    |- casmi2016
      |- MoNA-export-CASMI_2016.sdf
    |- casmi2017
      |- CASMI-solutions.csv
      |- Chal1to45Summary.csv
      |- challenges-001-045-msms-mgf-20170908 (unzip challenges-001-045-msms-mgf-20170908.zip)
    |- embl
      |- MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf
```

2. Prepare CASMI 2016, CASMI 2017, and EMBL-MCF 2.0 datasets:

```bash
python casmi2mgf.py --raw_dir ./data/casmi/ \
                --mgf_dir ./data/ \
                --data_config_path ./config/fiddle_tcn_casmi.yml

python embl2mgf.py --raw_path ./data/origin/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
                --mgf_path ./data/embl_mcf_2.0.mgf \
                --data_config_path ./config/fiddle_tcn_embl.yml
```

3. Test FIDDLE on all testsets: 

```bash
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_092724.pt \
                --result_path ./result/fiddle_casmi16.csv 

python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_092724.pt \
                --result_path ./result/fiddle_casmi17.csv 

python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_orbitrap_092724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_orbitrap_092724.pt \
                --result_path ./result/fiddle_embl.csv 
```

4. The more detailed experiments commands are in `./experiments_ex_test.sh`. 

## TODO

- [ ] PyPI package
- [ ] Online platform

## Citation

TBA

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


