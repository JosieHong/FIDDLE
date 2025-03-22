# --------------------------
# Data preprocessing
# --------------------------
python prepare_msms.py --dataset mona --instrument_type qtof --config_path ./config/fiddle_tcn_demo.yml --pkl_dir ./data/demo_cl_pkl/

# --------------------------
# Train & Predict on DEMO
# --------------------------
# without contrastive learning
python -u train_tcn_gpus.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_demo.yml \
--checkpoint_path ./check_point/fiddle_tcn_demo_qtof.pt \
--result_path ./result/fiddle_tcn_demo_qtof.csv --device 6 7 

# without data augmentation
python -u train_tcn_gpus_cl.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_demo_wo_da.yml \
--checkpoint_path ./check_point/fiddle_tcn_demo_qtof_cl_wo_da.pt \
--result_path ./result/fiddle_tcn_demo_qtof_cl_wo_da.csv --device 6 7 

# with contrastive learning and data augmentation
python -u train_tcn_gpus_cl.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_demo.yml \
--checkpoint_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--result_path ./result/fiddle_tcn_demo_qtof_cl.csv --device 6 7 

# FIDDLES (fdr model)
python prepare_fdr.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--fdr_dir ./data/demo_cl_pkl/ \
--device 6 7
python train_fdr.py --train_data ./data/demo_cl_pkl/qtof_random_fdr_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_fdr_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_fdr_demo_qtof_cl.pt \
--device 6 7 

# FIDDLES (test)
python run_fiddle.py --test_data ./data/demo_cl_pkl/qtof_random_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--fdr_resume_path ./check_point/fiddle_fdr_demo_qtof_cl.pt \
--result_path ./result/fiddle_demo_qtof_cl.csv --device 6 7
