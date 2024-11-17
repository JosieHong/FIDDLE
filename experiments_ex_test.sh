# ----------------------------------------
# Experiments on external test datasets: 
# CASMI 2016, CASMI 2017, and, EMBL MCF 2.0
# ID: 092724
# ----------------------------------------
# I. Data preprocessing
# ----------------------------------------

# 1. QTOF --------------------------------
python prepare_msms_all.py \
--dataset agilent nist20 nist23 mona waters gnps \
--instrument_type qtof \
--config_path ./config/fiddle_tcn_qtof.yml \
--pkl_dir ./data/cl_pkl_0927/ \
--test_title_list ./data/qtof_test_title_list_0927.txt \
--maxmin_pick

# 2. Orbitrap -----------------------------
python prepare_msms_all.py \
--dataset nist20 nist23 mona gnps \
--instrument_type orbitrap \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--pkl_dir ./data/cl_pkl_0927/ \
--test_title_list ./data/orbitrap_test_title_list_0927.txt \
--maxmin_pick

# 3. CASMI 2016, 2017, 2022 ----------------
# (TODO: remove 2022 later)
python casmi2mgf.py --data_config_path ./config/fiddle_tcn_casmi.yml
# for BUDDY and SIRIUS Orbitrap
python mgf_instances.py --input_path ./data/casmi2016.mgf \
                        --output_dir ./data_instances/casmi2016_pre/ \
                        --log ./data_instances/casmi2016_log.csv
python mgf_instances.py --input_path ./data/casmi2017.mgf \
                        --output_dir ./data_instances/casmi2017_pre/ \
                        --log ./data_instances/casmi2017_log.csv
python mgf_instances.py --input_path ./data/casmi2022.mgf \
                        --output_dir ./data_instances/casmi2022_pre/ \
                        --log ./data_instances/casmi2022_log.csv

# 4. EMBL MCF 2.0 -------------------------
python embl2mgf.py --raw_path ./data/origin/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
                --mgf_path ./data/embl_mcf_2.0.mgf \
                --data_config_path ./config/fiddle_tcn_embl.yml
python mgf_instances.py --input_path ./data/embl_mcf_2.0.mgf \
                        --output_dir ./data_instances/embl_pre/ \
                        --log ./data_instances/embl_log.csv



# --------------------------
# II. Train on QTOF
# --------------------------

# FIDDLE
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_0927/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_0927/qtof_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_092724.pt \
--resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
--device 6 7 >> fiddle_tcn_qtof_092724.out 

# FIDDLES (fdr model)
python prepare_fdr.py \
--train_data ./data/cl_pkl_0927/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_0927/qtof_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
--fdr_dir ./data/cl_pkl_0927/ \
--device 4 5
nohup python train_fdr.py \
--train_data ./data/cl_pkl_0927/qtof_maxmin_fdr_train.pkl \
--test_data ./data/cl_pkl_0927/qtof_maxmin_fdr_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_fdr_qtof_092724.pt \
--device 4 5 > fiddle_fdr_qtof_092724.out



# --------------------------
# III. Train on Qrbitrap
# --------------------------
# FIDDLE
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_0927/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_0927/orbitrap_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_092724.pt \
--resume_path ./check_point/fiddle_tcn_orbitrap_092724.pt \
--device 4 5 >> fiddle_tcn_orbitrap_092724.out 

# FIDDLES (fdr model)
python prepare_fdr.py \
--train_data ./data/cl_pkl_0927/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_0927/orbitrap_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_092724.pt \
--fdr_dir ./data/cl_pkl_0927/ \
--device 4 5 
python train_fdr.py \
--train_data ./data/cl_pkl_0927/orbitrap_maxmin_fdr_train.pkl \
--test_data ./data/cl_pkl_0927/orbitrap_maxmin_fdr_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_092724.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_fdr_orbitrap_092724.pt \
--device 4 5 



# ----------------------------------------
# IV. test on CASMI
# ----------------------------------------
# FIDDLE
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

# SIRIUS: 5.8.6 (2024-01-27)
# installation: 
# wget https://github.com/boecker-lab/sirius/releases/download/v5.8.6/sirius-5.8.6-linux64.zip
# unzip
# add bin into path
python run_sirius.py --instrument_type qtof --input_dir ./data_instances/casmi2016_pre/ \
                        --output_dir ./run_sirius/casmi2016_sirius_output/ \
                        --summary_dir ./run_sirius/casmi2016_sirius_summary/ \
                        --input_log ./data_instances/casmi2016_log.csv \
                        --output_log_dir ./run_sirius/casmi2016_sirius_log/ \
                        --output_log ./run_sirius/sirius_casmi2016.csv

python run_sirius.py --instrument_type qtof --input_dir ./data_instances/casmi2017_pre/ \
                        --output_dir ./run_sirius/casmi2017_sirius_output/ \
                        --summary_dir ./run_sirius/casmi2017_sirius_summary/ \
                        --input_log ./data_instances/casmi2017_log.csv \
                        --output_log_dir ./run_sirius/casmi2017_sirius_log/ \
                        --output_log ./run_sirius/sirius_casmi2017.csv

# BUDDY: 0.3.0
# installation: 
# conda create -n buddy python=3.9
# conda activate buddy
# pip install msbuddy
# conda activate buddy
python run_buddy.py --input_dir ./data_instances/casmi2016_pre/ \
                    --instrument_type qtof --top_k 10 \
                    --result_path ./run_buddy/buddy_casmi2016.csv 

python run_buddy.py --input_dir ./data_instances/casmi2017_pre/ \
                    --instrument_type qtof --top_k 10 \
                    --result_path ./run_buddy/buddy_casmi2017.csv 

# FIDDLE + BUDDY
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                    --config_path ./config/fiddle_tcn_qtof.yml \
                    --resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
                    --fdr_resume_path ./check_point/fiddle_fdr_qtof_092724.pt \
                    --buddy_path ./run_buddy/buddy_casmi2016.csv \
                    --result_path ./result/two_casmi16.csv 

python run_fiddle.py --test_data ./data/casmi2017.mgf \
                    --config_path ./config/fiddle_tcn_qtof.yml \
                    --resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
                    --fdr_resume_path ./check_point/fiddle_fdr_qtof_092724.pt \
                    --buddy_path ./run_buddy/buddy_casmi2017.csv \
                    --result_path ./result/two_casmi17.csv 



# ----------------------------------------
# V. test on EMBL MCF 2.0
# ----------------------------------------
# FIDDLE
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_092724.pt \
                --result_path ./result/fiddle_embl.csv 
# BUDDY
python run_buddy.py --input_dir ./data_instances/embl_pre/ \
                    --instrument_type qtof --top_k 5 \
                    --result_path ./run_buddy/buddy_embl.csv 

# SIURUS
python run_sirius.py --instrument_type qtof --input_dir ./data_instances/embl_pre/ \
                        --output_dir ./run_sirius/embl_sirius_output/ \
                        --summary_dir ./run_sirius/embl_sirius_summary/ \
                        --input_log ./data_instances/embl_log.csv \
                        --output_log_dir ./run_sirius/embl_sirius_log/ \
                        --output_log ./run_sirius/sirius_embl.csv

# FIDDLE + BUDDY
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_092724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_092724.pt \
                --buddy_path ./run_buddy/buddy_embl.csv \
                --result_path ./result/two_embl.csv 

