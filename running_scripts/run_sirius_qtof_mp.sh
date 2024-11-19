#!/bin/bash

# Define the start and end indices for each job
start_indices=(0 10000 20000 30000 40000)
end_indices=(9999 19999 29999 39999 41463)

# # TODO: correct the index ranges
# start_indices=(2500 10000 20000 30000 40000)
# end_indices=(9999 19999 29999 39999 41463)

# Loop through the start and end indices to run the jobs in parallel
for i in ${!start_indices[@]}; do
    start_index=${start_indices[$i]}
    end_index=${end_indices[$i]}

    # Run the job in the background
    nohup python -u run_sirius_mp.py --instrument_type qtof \
    --input_dir ./data_instances/qtof_pre_1007/ \
    --output_dir ./run_sirius_1007/qtof_sirius_output/ \
    --summary_dir ./run_sirius_1007/qtof_sirius_summary/ \
    --output_log_dir ./run_sirius_1007/qtof_sirius_log/ \
    --input_log ./data_instances/qtof_log_1007.csv \
    --start_index $start_index \
    --end_index $end_index \
    --output_log ./run_sirius_1007/sirius_qtof_test_${start_index}-${end_index}.csv > sirius_qtof_1007_${start_index}-${end_index}.out 2>&1 &

done

# # Wait for all background jobs to complete
# wait

echo "All SIRIUS jobs have been submitted and completed."
