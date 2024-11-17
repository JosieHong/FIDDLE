import os
import argparse

import re
import pandas as pd
import numpy as np
import time

from utils import formula_to_dict

def add_adduct(formula, adduct):
	if not isinstance(formula, str): 
		return formula 
		
	# Parse the formula and count the atoms
	atom_counts = formula_to_dict(formula)
	
	# Parse the adduct and update the atom counts, excluding "M"
	adduct_atoms = re.findall(r'[A-Z][a-z]?\d*', adduct.replace('[M+', '').replace(']', ''))
	for atom_group in adduct_atoms:
		atom = re.findall(r'[A-Z][a-z]?', atom_group)[0]
		if atom != "M":
			count = re.findall(r'\d*', atom_group)
			if not count:
				count = 1
			else:
				count = int(count[0]) if count[0] else 1
			atom_counts[atom[0]] = atom_counts.get(atom[0], 0) + count

	# Reconstruct the formula
	new_formula = ''
	for atom, count in sorted(atom_counts.items()):
		new_formula += atom
		if count > 1:
			new_formula += str(count)

	return new_formula



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='Mass Spectra to formula (SIRIUS)')
	parser.add_argument('--instrument_type', type=str, required=True,
						help='Name of the configuration profile. Predefined profiles are: `default`, `qtof`, `orbitrap`, `fticr`. Default: default')
	parser.add_argument('--input_dir', type=str, required=True,
						help='Folder to input')
	parser.add_argument('--output_dir', type=str, required=True,
						help='Folder to output')
	parser.add_argument('--summary_dir', type=str, required=True,
						help='Folder to summarization')
	parser.add_argument('--start_index', type=int, required=True,
						help='File index to start')
	parser.add_argument('--end_index', type=int, required=True,
						help='File index to end')

	parser.add_argument('--input_log', type=str, required=True,
						help='Same path to the log of mgf_instances.py')
	parser.add_argument('--output_log', type=str, required=True,
						help='Path to the save the final results')
	parser.add_argument('--output_log_dir', type=str, required=True,
						help='Folder to output logs')
	args = parser.parse_args()

	origin_df = pd.read_csv(args.input_log, index_col=0)
	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(args.summary_dir, exist_ok=True)
	os.makedirs(args.output_log_dir, exist_ok=True)
	
	# 1. run sirius 
	file_list = os.listdir(args.input_dir)
	# only run the files in the range (start_index, end_index)
	assert args.start_index < args.end_index
	file_list = [f for f in file_list if f.endswith('.mgf') \
				and args.start_index <= int(f.replace('.mgf', '')) <= args.end_index]
	file_list.sort()

	log_check = args.start_index
	log_size = 100
	res_dict = {'SortedIndex': [], 'RunningTime': [], 
				'PredFormula_1': [], 'Adduct_1': [], 'SiriusScore_1': [],
				'PredFormula_2': [], 'Adduct_2': [], 'SiriusScore_2': [],
				'PredFormula_3': [], 'Adduct_3': [], 'SiriusScore_3': [],
				'PredFormula_4': [], 'Adduct_4': [], 'SiriusScore_4': [],
				'PredFormula_5': [], 'Adduct_5': [], 'SiriusScore_5': []}
	for file_name in file_list: 
		# set the output dirs
		output_idx = log_check // log_size
		output_log = str(args.input_log.split('/')[-1]).replace('.csv', '_wtime_{}.csv'.format(output_idx))
		output_dir = os.path.join(args.output_dir, str(log_check))
		summary_dir = os.path.join(args.summary_dir, str(log_check))
		
		start_time = time.time()
		os.system("sirius --input {} \
					--output {} \
					config --IsotopeSettings.filter=true \
					--FormulaSearchDB= \
					--Timeout.secondsPerTree=0 \
					--FormulaSettings.enforced=HCNOP \
					--Timeout.secondsPerInstance=300 \
					--AdductSettings.detectable=[[M-H4O2+H]+,[M-H2O-H]-,[M+H]+,[M-H]-,[M-H2O+H]+] \
					--UseHeuristic.mzToUseHeuristicOnly=650 \
					--AlgorithmProfile={} \
					--IsotopeMs2Settings=IGNORE \
					--MS2MassDeviation.allowedMassDeviation=10.0ppm \
					--NumberOfCandidatesPerIon=1 \
					--UseHeuristic.mzToUseHeuristic=300 \
					--FormulaSettings.detectable=B,Cl,Br,S,F,I \
					--NumberOfCandidates=5 \
					--AdductSettings.fallback=[[M-H4O2+H]+,[M-H2O-H]-,[M+H]+,[M-H]-,[M-H2O+H]+] \
					--RecomputeResults=false formula --profile {} write-summaries --output {}".format(os.path.join(args.input_dir, file_name), output_dir, args.instrument_type, args.instrument_type, summary_dir))
		running_time = time.time() - start_time
		
		# 1.1 calculate sorted index
		sorted_index = file_name.replace('.mgf', '').lstrip('0')
		if sorted_index == '':
			sorted_index = 0
		else: 
			sorted_index = int(sorted_index)
		res_dict['SortedIndex'].append(sorted_index)
		res_dict['RunningTime'].append(running_time)

		# 1.2 get the predicted formula
		instance_list = [f for f in os.listdir(output_dir) if not f.startswith('.')]
		tmp_formulas = []
		tmp_scores = []
		tmp_adducts = []
		for inst in instance_list:
			# extract only predicted formula
			full_path = os.path.join(summary_dir, inst, 'formula_candidates.tsv')
			try: 
				df = pd.read_csv(full_path, sep='\t')
				tmp_formulas.append(df.loc[0, 'molecularFormula']) # somehow molecular formula doesn't contain adduct
				tmp_scores.append(float(df.loc[0, 'SiriusScore']))
				tmp_adducts.append(df.loc[0, 'adduct'])
			except FileNotFoundError as e:
				print(f"File not found: {e.filename}") 
				continue
			
		# 1.3 rank the formula by score
		top_idx = np.argsort(tmp_scores)[-3:][::-1]

		# 1.4 update the records
		for i in range(5): 
			if i < len(top_idx): 
				res_dict[f'PredFormula_{i+1}'].append(tmp_formulas[top_idx[i]])
				res_dict[f'Adduct_{i+1}'].append(tmp_adducts[top_idx[i]])
				res_dict[f'SiriusScore_{i+1}'].append(tmp_scores[top_idx[i]])
			else:
				res_dict[f'PredFormula_{i+1}'].append(np.NaN)
				res_dict[f'Adduct_{i+1}'].append(np.NaN)
				res_dict[f'SiriusScore_{i+1}'].append(np.NaN)

		# 2. write log in every itteration
		log_check += 1
		if log_check % log_size == 0: 
			res_df = pd.DataFrame.from_dict(res_dict)
			out_df = origin_df.merge(res_df, on='SortedIndex')
		
			out_df.to_csv(os.path.join(args.output_log_dir, output_log))
			print('Saved {}. Init res_dict.'.format(os.path.join(args.output_log_dir, output_log)))
			res_dict = {'SortedIndex': [], 'RunningTime': [], 
						'PredFormula_1': [], 'Adduct_1': [],  'SiriusScore_1': [],
						'PredFormula_2': [], 'Adduct_2': [], 'SiriusScore_2': [],
						'PredFormula_3': [], 'Adduct_3': [], 'SiriusScore_3': [],
						'PredFormula_4': [], 'Adduct_4': [], 'SiriusScore_4': [],
						'PredFormula_5': [], 'Adduct_5': [], 'SiriusScore_5': []}

	# 3. write log in the end
	res_df = pd.DataFrame.from_dict(res_dict)
	out_df = origin_df.merge(res_df, on='SortedIndex')

	print(out_df)
	out_df.to_csv(os.path.join(args.output_log_dir, output_log))
	print('Saved {}. Done!'.format(os.path.join(args.output_log_dir, output_log)))

	# 4. merge all logs
	file_list = os.listdir(args.output_log_dir)
	file_list.sort()
	dfs = []
	for i, f in enumerate(file_list): 
		df = pd.read_csv(os.path.join(args.output_log_dir, f), index_col=0)
		dfs.append(df)
	dfs = pd.concat(dfs)
	dfs = dfs.sort_values(by=['SortedIndex'])
	print(dfs)
	dfs.to_csv(args.output_log)
	print('Save the concatened logs!')

	# 5. somehow molecular formula doesn't contain adduct, add it now
	dfs = pd.read_csv(args.output_log)
	for i in range(5): 
		dfs['FinFormula_{}'.format(i+1)] = dfs.apply(lambda x: add_adduct(x['PredFormula_{}'.format(i+1)],
																			x['Adduct_{}'.format(i+1)]), axis=1)
	print(dfs)
	dfs.to_csv(args.output_log.replace('.csv', '_modified.csv'))
	print('Save the concatened logs with modifications!')
