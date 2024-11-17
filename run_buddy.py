import os
import argparse
import time
import pandas as pd

from msbuddy import Msbuddy, MsbuddyConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mass Spectra to formula (Msbuddy)')
    parser.add_argument('--instrument_type', type=str, default='default',
                        help='Name of the configuration profile. Predefined profiles are: `default`, `qtof`, `orbitrap`, `fticr`. Default: default')
    parser.add_argument('--top_k', type=int, required=True,
                        help='Top k formula to output')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder to input')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to output')
    args = parser.parse_args()

    msb_config = MsbuddyConfig(ms_instr=args.instrument_type,
                               ppm=True,
                               ms1_tol=5,
                               ms2_tol=10,
                               halogen=False)

    msb_engine = Msbuddy(msb_config)

    mgf_list = [f for f in os.listdir(args.input_dir) if f.endswith('.mgf')]
    res_dfs = []
    for mgf_file in mgf_list:
        start_time = time.time()
        msb_engine.load_mgf(os.path.join(args.input_dir, mgf_file))
        try:
            msb_engine.annotate_formula()
        except:
            print(f'Error in {mgf_file}')
            continue
        running_time = time.time() - start_time

        res = msb_engine.get_summary()
        for j, meta_feature in enumerate(msb_engine.data):
            if meta_feature.candidate_formula_list == None:
                res[j][f'formula_rank_{i+1}'] = None
                res[j][f'estimated_fdr_{i+1}'] = None
            else: 
                for i, candidate in enumerate(meta_feature.candidate_formula_list):
                    if i >= args.top_k:
                        break
                    res[j][f'formula_rank_{i+1}'] = str(candidate.formula)
                    res[j][f'estimated_fdr_{i+1}'] = str(candidate.estimated_fdr)

        res_df = pd.DataFrame(res)
        res_df['Running Time'] = running_time
        res_dfs.append(res_df)

    final_df = pd.concat(res_dfs, ignore_index=True)
    final_df.to_csv(args.result_path)
    print(f'Save the results to {args.result_path}')
