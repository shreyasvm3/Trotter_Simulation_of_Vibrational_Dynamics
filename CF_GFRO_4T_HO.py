import h5py
import numpy as np
import time
from sys import exit, argv

def read_ham_tensors(mol, nmodals, is_loc = True, num_couplings = 2, dip_anh = True, **kwargs):

	if is_loc:
		datafile = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/{mol}_loc.hdf5', 'r')
	else:
		datafile = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/{mol}.hdf5', 'r')
	
	#Read in Hamiltonian generated from taylor expanded potential
	H1 = np.array(datafile['T1'][()])
	H1 = H1[:,:nmodals,:nmodals]

	if num_couplings > 1:
		H2 = np.array(datafile['T2'][()])
		H2 = H2[:,:,:nmodals,:nmodals,:nmodals,:nmodals]
	if num_couplings > 2:
		H3 = np.array(datafile['T3'][()])
		H3 = H3[:,:,:,:nmodals,:nmodals,:nmodals,:nmodals,:nmodals,:nmodals]

	D1 = np.array(datafile['taylor_D1'][()])
	D1 = D1[:,:nmodals,:nmodals]

	if dip_anh:
		D2 = np.array(datafile['taylor_D2'][()])
		D2 = D2[:,:,:nmodals,:nmodals,:nmodals,:nmodals]
		D_arr = [D1, D2]
	else:
		D2 = None
		D_arr = [D1]
	
	if num_couplings == 2:
		H_arr = [H1, H2]
	elif num_couplings == 3:
		H_arr = [H1, H2, H3]
 
	return H_arr, D_arr


t0 = time.time()
#Define the molecule
mol = str(argv[1])
print(f'Molecules is: {mol}')

au_to_cm = 219474.63

nmodals = int(argv[2])
print(f'Number of modals is: {nmodals}')

is_loc = input-loc #whether Hamiltonian comes from localized modals
num_couplings = input-num-coup #number of mode couplings for the Hamiltonian

H_arr, _ = read_ham_tensors(mol, nmodals, is_loc = is_loc, num_couplings = num_couplings)

H1 = H_arr[0] * au_to_cm
if num_couplings == 2:
	from vibrant.algorithm.trotter.GFRO_utils_2MC import load_params, save_params, get_fragment, obtain_gfro_fragment
	H2 = H_arr[1] * au_to_cm
elif num_couplings == 3:
	from vibrant.algorithm.trotter.GFRO_utils_3MC import load_params, save_params, get_fragment, obtain_gfro_fragment
	H2 = H_arr[1] * au_to_cm
	H3 = H_arr[2] * au_to_cm
else:
	ValueError('num_couplings should be in [2,3]')
nmodes  = H1.shape[0]

resume_flag = int(argv[3])
if resume_flag == 0:
	num_fragments = 0
	norms = []
	params = []
elif resume_flag == 1:
	print('Resuming Fragmentation')
	# Load in existing fragments:
	params, norms = load_params(mol, nmodals)

	#Store number of fragments already found
	num_fragments = len(params)

	# subtract all fragments from two/three_mode_term
	for x in params:
		if num_couplings == 2:
			H2f, H3f = get_fragment(x, nmodals, nmodes)
			H2 -= H2f

		elif num_couplings == 3:
			H2f, H3f = get_fragment(x, nmodals, nmodes)
			H2 -= H2f
			H3 -= H3f
        
else:
    ValueError('Resume_flag must be 0 or 1')

# get current norm
if num_couplings == 2:
	current_norm = np.sum(H2 * H2) 
elif num_couplings == 3:
	current_norm = np.sum(H2 * H2) + np.sum(H3 * H3)

norms.append(current_norm)

with open('Fragments.out', 'a') as f:
	print(f'Initial Norm : {current_norm}\n', file=f)

t1 = time.time()
with open('Fragments.out', 'a') as f:
	print(f'Time taken for setup (in mins): {(t1-t0)/60.0}', file=f)   

t0 = t1
# do GFRO decomposition with convergence criterion tol=50
numtag        = 1000
tol           = 50

for k in range(numtag):
	# end decomposition if remaining fraction of norm is less than tol
	if current_norm < tol:
		break
	#print('At fragment #',k)

	# obtain fragment parameters and remove fragment from H2/H3
	if num_couplings == 2:
		sol = obtain_gfro_fragment(H2)
		H2f = get_fragment(sol.x, nmodals, nmodes)
		H2 -= H2f

	elif num_couplings == 3:
		sol = obtain_gfro_fragment(H2,H3)
		H2f, H3f = get_fragment(sol.x, nmodals, nmodes)
		H2 -= H2f
		H3 -= H3f
    
	# save current parameters and current norm
	params.append(sol.x)

	if num_couplings == 2:
		current_norm = np.sum(H2 * H2) 
	elif num_couplings == 3:
		current_norm = np.sum(H2 * H2) + np.sum(H3 * H3)

	norms.append(current_norm)

	# print output
	num_fragments += 1
	with open('Fragments.out', 'a') as f:
		print(f'fragment count : {num_fragments}, current norm : {np.round(current_norm, 15)}\n', file=f)

	save_params(mol, nmodals, params, norms)
    
	t1 = time.time()
	with open('Fragments.out', 'a') as f:
		print(f'Time taken for this fragment (in mins): {(t1-t0)/60.0}', file=f)   
	t0 = t1 
