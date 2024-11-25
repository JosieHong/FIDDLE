#!/bin/bash

# Define the start and end indices for each job
start_indices=(0 100000 150000 200000 250000 300000 350000 400000)
end_indices=(99999 149999 199999 249999 299999 349999 399999 408589)

# Loop through the start and end indices to run the jobs in parallel
for i in ${!start_indices[@]}; do
    start_index=${start_indices[$i]}
    end_index=${end_indices[$i]}

    # Run the job in the background
    nohup python -u run_sirius_mp.py --instrument_type orbitrap \
    --input_dir ./data_instances/orbitrap_pre_1007/ \
    --output_dir ./run_sirius_1007/orbitrap_sirius_output/ \
    --summary_dir ./run_sirius_1007/orbitrap_sirius_summary/ \
    --output_log_dir ./run_sirius_1007/orbitrap_sirius_log/ \
    --input_log ./data_instances/orbitrap_log_1007.csv \
    --start_index $start_index \
    --end_index $end_index \
    --output_log ./run_sirius_1007/sirius_orbitrap_test_${start_index}-${end_index}.csv > sirius_orbitrap_1007_${start_index}-${end_index}.out 2>&1 &

    echo "Submitted SIRIUS job for indices $start_index to $end_index"
done

# # Wait for all background jobs to complete
# wait

echo "All SIRIUS jobs have been submitted and completed."
