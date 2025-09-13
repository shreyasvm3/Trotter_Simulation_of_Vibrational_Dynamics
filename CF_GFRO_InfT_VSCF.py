import h5py
import numpy as np
import time
from sys import exit, argv
from vibrant.vscf import VSCF_Fock


def create_VSCF_tensors(mol, nmodals, is_loc = True, num_couplings = 2, dip_anh = True, save_flag = False, **kwargs):

	#Read in tensors in HO basis
	if is_loc:
		f = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/{mol}_loc.hdf5', 'r')
	else:
		f = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/{mol}.hdf5', 'r')
    
	if num_couplings not in [2,3]:
		ValueError('Num_couplings should be in [2,3]')

	H1_HO = f['H1'][()]
	if num_couplings > 1:
		H2_HO = f['H2'][()]
	if num_couplings > 2:
		H3_HO = f['H3'][()]

	D1_HO = f['D1'][()]
	if dip_anh:
		D2_HO = f['D2'][()]
		dip_arr = [D1_HO, D2_HO]
	else:	
		D2 = None
		dip_arr = [D1_HO]	
	f.close()

	if num_couplings == 1:
		ham_arr = [H1_HO]
	elif num_couplings == 2:
		ham_arr = [H1_HO,H2_HO]
	elif num_couplings == 3:
		ham_arr = [H1_HO,H2_HO,H3_HO]

	#Perform VSCF calculation
	fock_object = VSCF_Fock(ham_arr)
	fock_object.VSCF(verbose = True)

	modals = fock_object.M * [nmodals] 

	Hvscf, Dvscf = fock_object.vscf_objects(dip_arr, modals)

	if save_flag:
		#Save VSCF tensors
		if is_loc:
    			f = h5py.File(f'{mol}_VSCF_{num_couplings}MC_loc.hdf5', 'w')
		else:
    			f = h5py.File(f'{mol}_VSCF_{num_couplings}MC.hdf5', 'w')

		f.create_dataset('H1',data = Hvscf[0])
		if num_couplings > 1 :
			f.create_dataset('H2',data = Hvscf[1])
		if num_couplings > 2:
			f.create_dataset('H3',data = Hvscf[2])

		f.create_dataset('D1',data = Dvscf[0])
		if dip_anh:
			f.create_dataset('D2',data = Dvscf[1])
		f.close()

	return Hvscf, Dvscf

def read_vscf_tensors(mol, nmodals, is_loc = True, num_couplings = 2, dip_anh = True, **kwargs):

	if is_loc:
		datafile = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/VSCF_tensors/{mol}_VSCF_{nmodals}M_{num_couplings}MC_loc.hdf5', 'r')
	else:
		datafile = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/VSCF_tensors/{mol}_VSCF_{nmodals}M_{num_couplings}MC.hdf5', 'r')
	
	#Read in Hamiltonian
	H1 = np.array(datafile['H1'][()])
	H1 = H1[:,:nmodals,:nmodals]

	if num_couplings > 1:
		H2 = np.array(datafile['H2'][()])
		H2 = H2[:,:,:nmodals,:nmodals,:nmodals,:nmodals]
	if num_couplings > 2:
		H3 = np.array(datafile['H3'][()])
		H3 = H3[:,:,:,:nmodals,:nmodals,:nmodals,:nmodals,:nmodals,:nmodals]

	D1 = np.array(datafile['D1'][()])
	D1 = D1[:,:nmodals,:nmodals]

	if dip_anh:
		D2 = np.array(datafile['D2'][()])
		D2 = D2[:,:,:nmodals,:nmodals,:nmodals,:nmodals]
		Dvscf = [D1, D2]
	else:
		D2 = None
		Dvscf = [D1]
	
	if num_couplings == 2:
		Hvscf = [H1, H2]
	elif num_couplings == 3:
		Hvscf = [H1, H2, H3]
 
	return Hvscf, Dvscf


def generate_ham_tensors(mol, nmodals, vscf_flag = 0, **kwargs):
	'''
	If vscf_flag = 0, Ham tensors in VSCF basis are read in from a stored file.
	If vscf_flag = 1, Ham tensors in HO basis are read in from stored file. 
	VSCF is then performed, and rotated tensors in VSCF basis are obtained.
	'''

	if vscf_flag == 0:
		Hvscf, Dvscf = read_vscf_tensors(mol, nmodals, **kwargs)

	elif vscf_flag == 1:
		Hvscf, Dvscf = create_VSCF_tensors(mol, nmodals, **kwargs)
	else:
		ValueError('vscf_flag should be 0 or 1.')

	return Hvscf, Dvscf



t0 = time.time()
#Define the molecule
mol = str(argv[1])
print(f'Molecules is: {mol}')

#Flag for VSCF. 
vscf_flag = int(argv[2])
au_to_cm = 219474.63

nmodals = int(argv[3])
print(f'Number of modals is: {nmodals}')

is_loc = input-loc #whether Hamiltonian comes from localized modals
num_couplings = input-num-coup #number of mode couplings for the Hamiltonian

Hvscf, _ = generate_ham_tensors(mol, nmodals, vscf_flag = vscf_flag, is_loc = is_loc, num_couplings = num_couplings, save_flag = True)

H1 = Hvscf[0] * au_to_cm
if num_couplings == 2:
	from vibrant.algorithm.trotter.GFRO_utils_2MC import load_params, save_params, get_fragment, obtain_gfro_fragment
	H2 = Hvscf[1] * au_to_cm
elif num_couplings == 3:
	from vibrant.algorithm.trotter.GFRO_utils_3MC import load_params, save_params, get_fragment, obtain_gfro_fragment
	H2 = Hvscf[1] * au_to_cm
	H3 = Hvscf[2] * au_to_cm
else:
	ValueError('num_couplings should be in [2,3]')
nmodes  = H1.shape[0]

resume_flag = int(argv[4])
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
