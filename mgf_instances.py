'''
aims to separate mgf file into one spectrum one file 
the output files are ordered by molecular isotopic mass
'''
import os 
import argparse

import pandas as pd
from pyteomics import mgf

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from molmass import Formula

from utils import MSLEVEL_MAP



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='Mass Spectra to formula (SIRIUS)')
	parser.add_argument('--input_path', type=str, required=True,
						help='Path to input (.mgf)')
	parser.add_argument('--output_dir', type=str, required=True,
						help='Folder to output')
	
	parser.add_argument('--log', type=str, required=True,
						help='Path to save the mapping between spectrum and assigned names')
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	
	# 1. load the mgf items into a dict -> pandas df
	mass_dict = {'SpecIndex': [], 'Mass': [], 'SMILES': [], 'Title': []}
	supp = mgf.read(args.input_path)
	print('Load {} data from {}'.format(len(supp), args.input_path))
	for idx, spec in enumerate(supp):
		title = spec['params']['title']
		s = spec['params']['smiles']
		f = CalcMolFormula(Chem.MolFromSmiles(s))
		try: 
			f = Formula(f)
			iso_mass = f.isotope.mass
		except: 
			continue

		mass_dict['SpecIndex'].append(idx)
		mass_dict['Mass'].append(iso_mass)
		mass_dict['SMILES'].append(s)
		mass_dict['Title'].append(title)

	mass_df = pd.DataFrame.from_dict(mass_dict)

	# 2. assign them files names
	mass_df = mass_df.sort_values(by=['Mass'])
	mass_df.reset_index(inplace=True, drop=True)
	mass_df.reset_index(inplace=True)
	mass_df = mass_df.rename(columns = {'index': 'SortedIndex'})

	mass_df.to_csv(args.log) # save the log

	assigned_index = pd.Series(mass_df.SortedIndex.values, index=mass_df.SpecIndex.values).to_dict()
	# print(assigned_index)

	# 3. reload the mgf items and save them by assigned file names
	supp = mgf.read(args.input_path)
	print('Reload {} data from {}'.format(len(supp), args.input_path))
	for idx, spec in enumerate(supp): 
		spec_out = {'params': {'title': spec['params']['title'],
								'ion': spec['params']['precursor_type']}, 
					'm/z array': spec['m/z array'], 
					'intensity array': spec['intensity array']} # output spectrum

		# print(str(assigned_index[idx]).zfill(5)+'.mgf')
		if idx not in assigned_index.keys(): 
			continue

		# only output the params used in SIRIUS and BUDDY
		# https://github.com/boecker-lab/sirius/blob/36da5e7ade9411c51b09d3682aac029792c092e6/sirius_doc/manual/demo-data/mgf/laudanosine.mgf
		# https://github.com/Philipbear/msbuddy/blob/main/demo/input_file.mgf
		# https://github.com/boecker-lab/sirius/issues/115

		spec_out['params']['pepmass'] = float(spec['params']['simulated_precursor_mz']) 
		
		if 'ms_level' in spec['params'].keys(): 
			spec_out['params']['ms_level'] = MSLEVEL_MAP[spec['params']['ms_level']]
		else: 
			spec_out['params']['ms_level'] = 2
		
		if 'charge' in spec['params'].keys(): 
			spec_out['params']['charge'] = spec['params']['charge']
		else: 
			spec_out['params']['charge'] = 1

		mgf.write([spec_out], os.path.join(args.output_dir, 
									str(assigned_index[idx]).zfill(5)+'.mgf'), file_mode="w", write_charges=False)
		print('Save {}'.format(str(assigned_index[idx]).zfill(5)+'.mgf'))
		