# ----------------------------------------
# Experiments on internal test datasets: 
# unique compounds in NIST23
# ID: 100724
# ----------------------------------------
# I. Data preprocessing
# ----------------------------------------

# 1. QTOF --------------------------------
python prepare_msms_nist23.py \
--dataset agilent nist20 nist23 mona waters gnps \
--instrument_type qtof \
--config_path ./config/fiddle_tcn_qtof.yml \
--pkl_dir ./data/cl_pkl_1007/
# BUDDY and SIRIUS QTOF
python mgf_instances.py --input_path ./data/cl_pkl_1007/qtof_test.mgf \
--output_dir ./data_instances/qtof_pre_1007/ \
--log ./data_instances/qtof_log_1007.csv

# 2. Orbitrap -----------------------------
python prepare_msms_nist23.py \
--dataset nist20 nist23 mona gnps \
--instrument_type orbitrap \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--pkl_dir ./data/cl_pkl_1007/
# for BUDDY and SIRIUS Orbitrap
python mgf_instances.py --input_path ./data/cl_pkl_1007/orbitrap_test.mgf \
--output_dir ./data_instances/orbitrap_pre_1007/ \
--log ./data_instances/orbitrap_log_1007.csv



# --------------------------
# II. Train on QTOF
# --------------------------

# FIDDLE
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_1007/qtof_train.pkl \
--test_data ./data/cl_pkl_1007/qtof_test.pkl \
--additional_f_data ./data/additional_formula.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_100724.pt \
--result_path ./result/fiddle_tcn_qtof_100724.csv --device 4 5 > fiddle_tcn_qtof_100724.out

# FIDDLES (fdr model)
python prepare_fdr.py \
--train_data ./data/cl_pkl_1007/qtof_train.pkl \
--test_data ./data/cl_pkl_1007/qtof_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--fdr_dir ./data/cl_pkl_1007/ \
--device 4 5
nohup python train_fdr.py \
--train_data ./data/cl_pkl_1007/qtof_fdr_train.pkl \
--test_data ./data/cl_pkl_1007/qtof_fdr_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_fdr_qtof_100724.pt \
--device 4 5 > fiddle_fdr_qtof_100724.out



# --------------------------
# III. Train on Qrbitrap
# --------------------------
# FIDDLE
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_1007/orbitrap_train.pkl \
--test_data ./data/cl_pkl_1007/orbitrap_test.pkl \
--additional_f_data ./data/additional_formula.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--result_path ./result/fiddle_tcn_orbitrap_100724.csv --device 6 7 > fiddle_tcn_orbitrap_100724.out

# FIDDLES (fdr model)
nohup python prepare_fdr.py \
--train_data ./data/cl_pkl_1007/orbitrap_train.pkl \
--test_data ./data/cl_pkl_1007/orbitrap_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--fdr_dir ./data/cl_pkl_1007/ \
--device 4 5 
nohup python train_fdr.py --train_data ./data/cl_pkl_1007/orbitrap_fdr_train.pkl \
--test_data ./data/cl_pkl_1007/orbitrap_fdr_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--device 4 5 > fiddle_fdr_orbitrap_100724.out



# --------------------------
# IV. Test on QTOF
# --------------------------
# BUDDY
nohup python run_buddy.py --instrument_type qtof --top_k 5 \
--input_dir ./data_instances/qtof_pre_1007/ \
--result_path ./run_buddy_1007/buddy_qtof_test_1007.csv > buddy_qtof_1007.out

# SIRIUS
nohup python -u run_sirius.py --instrument_type qtof \
--input_dir ./data_instances/qtof_pre_1007/ \
--output_dir ./run_sirius_1007/qtof_sirius_output/ \
--summary_dir ./run_sirius_1007/qtof_sirius_summary/ \
--output_log_dir ./run_sirius_1007/qtof_sirius_log/ \
--input_log ./data_instances/qtof_log_1007.csv \
--output_log ./run_sirius_1007/sirius_qtof_test.csv > sirius_qtof_1007.out
# run SIRIUS in multiple processes
# bash run_sirius_qtof_mp.sh

# FIDDLES
python run_fiddle.py --test_data ./data/cl_pkl_1007/qtof_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
--result_path ./result/fiddle_qtof_100724.csv --device 5

# FIDDLE + BUDDY
nohup python run_fiddle.py --test_data ./data/cl_pkl_1007/qtof_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
--buddy_path ./run_buddy_1007/buddy_qtof_test_1007.csv \
--result_path ./result/two_qtof_test_100724.csv --device 5

# FIDDLE + BUDDY + SIRIUS
# python run_fiddle.py --test_data ./data/cl_pkl/qtof_maxmin_test.mgf \
# --config_path ./config/fiddle_tcn_qtof.yml \
# --resume_path ./check_point/fiddle_tcn_qtof_080624.pt \
# --fdr_resume_path ./check_point/fiddle_fdr_qtof_080624.pt \
# --buddy_path ./run_buddy/buddy_qtof_test.csv \
# --sirius_path ./run_sirius/sirius_qtof_test.csv \
# --result_path ./result/all_qtof_test.csv



# --------------------------
# V. Test on Orbitrap
# --------------------------
# BUDDY
nohup python run_buddy.py --instrument_type orbitrap --top_k 5 \
--input_dir ./data_instances/orbitrap_pre_1007/ \
--result_path ./run_buddy_1007/buddy_orbitrap_test_1007.csv > buddy_orbitrap_1007.out

# SIRIUS
nohup python -u run_sirius.py --instrument_type orbitrap \
--input_dir ./data_instances/orbitrap_pre_1007/ \
--output_dir ./run_sirius_1007/orbitrap_sirius_output/ \
--summary_dir ./run_sirius_1007/orbitrap_sirius_summary/ \
--output_log_dir ./run_sirius_1007/orbitrap_sirius_log/ \
--input_log ./data_instances/orbitrap_log_1007.csv \
--output_log ./run_sirius_1007/sirius_orbitrap_test_1007.csv > sirius_orbitrap_1007.out
# run SIRIUS in multiple processes
# bash run_sirius_orbitrap_mp.sh

# FIDDLES
nohup python run_fiddle.py --test_data ./data/cl_pkl_1007/orbitrap_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--fdr_resume_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--result_path ./result/fiddle_orbitrap_100724.csv --device 6

# FIDDLE + BUDDY
nohup python run_fiddle.py --test_data ./data/cl_pkl_1007/orbitrap_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--fdr_resume_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--buddy_path ./run_buddy_1007/buddy_orbitrap_test_1007.csv \
--result_path ./result/two_orbitrap_test_100724.csv --device 4



# --------------------------
# VI. Test on CASMI and EMBL
# --------------------------
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --result_path ./result/fiddle_casmi16_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2016.csv \
                --result_path ./result/two_casmi16_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2016.csv \
                --sirius_path ./run_sirius/sirius_casmi2016.csv \
                --result_path ./result/all_casmi16_exnist23.csv 

python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --result_path ./result/fiddle_casmi17_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2017.csv \
                --result_path ./result/two_casmi17_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2017.csv \
                --sirius_path ./run_sirius/sirius_casmi2017.csv \
                --result_path ./result/all_casmi17_exnist23.csv 

python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --result_path ./result/fiddle_embl_exnist23.csv 
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_embl.csv \
                --result_path ./result/two_embl_exnist23.csv 
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --fdr_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_embl.csv \
                --sirius_path ./run_sirius/sirius_embl.csv \
                --result_path ./result/all_embl_exnist23.csv 