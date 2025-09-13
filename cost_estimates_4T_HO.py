import h5py
import numpy as np
from mpi4py import MPI
import pickle
import sys
from vibrant.vscf import VSCF_Fock
from vibrant.algorithm.lt_qpe import dipole_based_normalizer
from vibrant.qubit_utils import vib_from_modes
from vibrant.system_prep.initial_state import dipole_to_vector
from vibrant.algorithm.simulate import post_process, run_mpi_generic
from vibrant.algorithm.estimates import count_steps_fixed_time, rz_to_t_gates, runtime_from_rz
from vibrant.algorithm.trotter.cform_error import r_from_pt
from time import time
import matplotlib.pyplot as plt
import os
import itertools
np.set_printoptions(precision=4)
np.set_printoptions(formatter={'float': '{:.2e}'.format})
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()

'''
INPUTS FOR SIMULATOR:
python run_LT.py mol_name num_modals_min num_modals_max num_couplings is_loc r_arr (k_arr) ((shots_arr))
arguments in parenthesis are optional

e.g.
mol_name = H2S
num_modals_min = 2
num_modals_max = 5
num_couplings = 2
is_loc = True

example 1:
r_arr = 0,_,1

will calculate 3 spectra, all with (default) kmax=None, (infinite) shots=None and r corresponding to:
	- (r=0) exact time evolution, 
	- (r=N) r=None: perturbation theory bound
	- r=1

example 2:
r_arr = _,_
k_arr = _,100

second spectra will be ran using kmax=100
'''
data_folder = ''

pl_device = "lightning.kokkos" #pennylane device
dip_anh = True #whether anharmonic components of dipole are included
epsilon_in_cm = 7 #eigenvalue accuracy for PT bound
trotter_order = 2 #order of Trotter expansion
#kmax_epsilon   = 1e-3  # epsilon to decide max evolution time.
rz_epsilon = 1e-10  #to what accuracy should Rz gates be implemented, determines number of T-gates per controlled Rz
clock_rate = 1e6
frag_method = 'input-frag-method'
frag_tol    = input-frag-tol
tau  = 250  # (in a.u.) this is one time-step. We have total of kmax such steps for the full propagation.
# tau will be divided into r smaller steps of tau/r for Trotter propagation.
dark_plots = False

au_to_cm = 219475
#eta = 10

mol_name = sys.argv[1]
num_modals_min = int(sys.argv[2])
num_modals_max = int(sys.argv[3])
if int(sys.argv[4]) in (1,2,3):
        num_couplings = int(sys.argv[4])
else:
        raise ValueError(f"Entered num_coupling other than accepted values: 1, 2 and 3")
#whether Hamiltonian comes from localized modals
if int(sys.argv[5]) == 0:
   is_loc = False 
elif int(sys.argv[5]) == 1:
   is_loc = True 
else: 
    raise ValueError(f"Enter 0 or 1 for is_loc. Neither was supplied.")
if is_loc and num_couplings == 3:
        raise ValueError(f"Localized Hamiltonians only supported with 1 & 2 mode couplings. 3 mode couplings included here.")

r_arr_string = sys.argv[6].split(',')
num_rs = len(r_arr_string)
r_arr = []
for r_string in r_arr_string:
	if r_string == '_':
		r_arr.append(None)
	else:
		r_arr.append(int(r_string))
if rank == 0:
	print(f"Using array of r's for simulations {r_arr}")

if len(sys.argv) > 7:
	k_arr_string = sys.argv[7].split(',')
	has_k = True
else:
	has_k = False

if has_k == False:
	k_arr = num_rs * [None]
else:
	num_ks = len(k_arr_string)
	if num_ks == 1:
		if k_arr_string[0] == '_':
			k_arr = num_rs * [None]
		else:
			k_arr = num_rs * [int(k_arr_string[0])]
	else:
		if num_ks != num_rs:
			raise ValueError(f"Entered array of k's but has different length from array of r's!")
		k_arr = []
		for k_string in k_arr_string:
			if k_string == '_':
				k_arr.append(None)
			else:
				k_arr.append(int(k_string))
if rank == 0:
	print(f"Using array of k's for simulations {k_arr}")

if len(sys.argv) > 8:
	shots_arr_string = sys.argv[8].split(',')
	has_shots = True
else:
	has_shots = False

if has_shots == False:
	shots_arr = num_rs * [None]
else:
	num_shots = len(shots_arr_string)
	if num_shots == 1:
		if shots_arr_string[0] == '_':
			shots_arr = num_rs * [None]
		else:
			shots_arr = num_rs * [int(shots_arr_string[0])]
	else:
		if num_shots != num_rs:
			raise ValueError(f"Entered array of shots's but has different length from array of r's!")
		shots_arr = []
		for shots_string in shots_arr_string:
			if shots_string == '_':
				shots_arr.append(None)
			else:
				shots_arr.append(int(shots_string))
if rank == 0:
	print(f"Using array of shots's for simulations {shots_arr}")
	
TIMES = [time()]

if dark_plots == True:
	plt.style.use('dark_background')

if is_loc:
    fname = rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/{mol_name}_loc.hdf5'

else:
    fname = rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/{mol_name}.hdf5'
 
H1_full = None
H2_full = None
H3_full = None
D1_full = None
D2_full = None

H1 = None
H2 = None
H3 = None
D1 = None
D2 = None

if rank == 0:
	f = h5py.File(fname, 'r')
	H1_full = f['T1'][()]
	if num_couplings > 1:
		H2_full = f['T2'][()]
	if num_couplings > 2:
		H3_full = f['T3'][()]

	D1_full = f['taylor_D1'][()]
	if dip_anh:
		D2_full = f['taylor_D2'][()]
	else:
		D2_full = None
	f.close()


modals_arr = np.arange(num_modals_min,num_modals_max + 1)

for num_modals in modals_arr:
	if rank == 0:
		print(f"Molecule name is {mol_name}")
		print(f"Number of modals per mode is {num_modals}")
		print(f"Running with PennyLane device {pl_device}")
		print(f"Mode localization is set to {is_loc}")
		print(f"Number of mode couplings included is {num_couplings}")
		print(f"Anharmonic dipole components inclusion is set to {dip_anh}")

		H1 = H1_full[:,:num_modals,:num_modals]
		D1 = D1_full[:,:num_modals,:num_modals]
		if num_couplings > 1:
			H2 = H2_full[:,:,:num_modals,:num_modals,:num_modals,:num_modals]
		if num_couplings > 2:
			H3 = H3_full[:,:,:,:num_modals,:num_modals,:num_modals,:num_modals,:num_modals,:num_modals]
		if dip_anh:
			D2 = D2_full[:,:,:num_modals,:num_modals,:num_modals,:num_modals]

	H1 = np.array(COMM.bcast(H1, root=0))
	D1 = np.array(COMM.bcast(D1, root=0))
	if dip_anh:
		D2 = np.array(COMM.bcast(D2, root=0))
		dip_arr = [D1, D2]
	else:
		dip_arr = [D1]

	if num_couplings == 1:
		ham_arr = [H1]
	elif num_couplings == 2:
		H2 = np.array(COMM.bcast(H2, root=0))
		ham_arr = [H1,H2]
	elif num_couplings == 3:
		H2 = np.array(COMM.bcast(H2, root=0))
		H3 = np.array(COMM.bcast(H3, root=0))
		ham_arr = [H1,H2,H3]

	fock_object = VSCF_Fock(ham_arr)
	#Don't want to do VSCF:
	#fock_object.VSCF(verbose=False)
	#vscf_energy = fock_object.fock_energy()

	modals = fock_object.modals
	#Hnorm = dipole_based_normalizer(fock_object, dip_arr, modals, verbose=False)
	#Don't need to calculate Hnorm. It will be Hnorm = 2*pi/tau. The variable Hnorm is actually Hnorm/4 = pi/2/tau
	Hnorm = np.pi / 2 / tau
	freq_max = 0.99 * 4 * Hnorm * au_to_cm  # 4 * Hnorm is the actual Hnorm
	if rank == 0:
		print(f"Norm of Hamiltonian is {Hnorm:.2e}, corresponding to spectral window up to {freq_max:.0f} cm-1")
	TIMES.append(time())
	if rank == 0:
		print(f"Time elapsed for doing VSCF and finding norm was {TIMES[-1]-TIMES[-2]:.2f} seconds")

	my_r = r_arr[0]
	if my_r is None:
		rpt = r_from_pt(fock_object, dip_arr, modals, Hnorm, frag_method = frag_method, vscf_flag = False, epsilon_in_cm=epsilon_in_cm, verbose=False, mol_name = mol_name, is_loc = is_loc, num_couplings = num_couplings, frag_tol = frag_tol)
		if rank == 0:
			print(f"Found r={rpt} using perturbation theory, bringing to integer {np.ceil(rpt)}")
			rpt = int(np.ceil(rpt))
	else:
		assert my_r != 0, "Trotter r was input as 0, should be non-zero for cost estimation"
		rpt = np.copy(my_r)

	rpt = COMM.bcast(rpt, root=0)

	if my_r is None:
		for num_sim in range(num_rs):
			r_arr[num_sim] = rpt

	TIMES.append(time())
	if rank == 0:
		print(f"Time elapsed for obtaining perturbative r was {TIMES[-1]-TIMES[-2]:.2f} seconds")

	au_max = freq_max / au_to_cm
	freqs = np.linspace(0,au_max,10000)

	#Hvscf, Dvscf = fock_object.vscf_objects(dip_arr, modals)
	num_qubits, pl_ham = vib_from_modes(ham_arr, modals, ps=True)
	
	'''
	if dip_anh == False:
		D2_vscf = None
	else:
		D2_vscf = Dvscf[1]
	'''

	TIMES.append(time())
	if rank == 0:
		print(f"Time elapsed for building VSCF matrices and PL operators was {TIMES[-1]-TIMES[-2]:.2f} seconds")
	time_index_before_spectra = len(TIMES) - 1

	if rank == 0:
		print('\n##############################')
		print("Starting cost calculations")
		print('##############################\n')

	if trotter_order == 1:
		method = 'trotter'
	elif trotter_order > 1:
		method = f'trotter{trotter_order}'
	if rank == 0:
		print(f'Trotter simulations will be ran with method {method}')

	t_count_list = []
	rz_per_trot_list = []
	trot_calls_list = []
	cost_arr = []
	#Calculate # of T-gates per R_z gate for specified R_z gate compilation error
	rz_synth_num = rz_to_t_gates(rz_epsilon, controlled = False)

	for sim_num in range(num_rs):
		my_r = r_arr[sim_num]
		if my_r is None:
			my_r = rpt
		my_kmax = k_arr[sim_num]
		my_shots = shots_arr[sim_num]
		if rank == 0:
			print(f'\n\nStarting cost calculation for r={my_r}')
		if my_r == 0:
			raise ValueError(f"r=0, need perturbative r to get cost estimate")

		#{ Added by SVM 
		trot_calls, tot_frags, rz_per_trot, rz_count = count_steps_fixed_time(pl_ham, my_r, my_kmax, trotter_order= trotter_order, frag_method = frag_method, num_couplings = num_couplings, is_loc = is_loc ,mol_name = mol_name, modals = modals, frag_tol = frag_tol, vscf_flag = False)
		runtime = runtime_from_rz(rz_count,rz_epsilon,clock_rate)
		t_count = rz_count * rz_synth_num
		#}

		TIMES.append(time())
		if rank == 0:
			print(f"Time elapsed for cost estimation {sim_num+1} out of {num_rs} was {TIMES[-1]-TIMES[-2]:.2f} seconds")
		rz_per_trot_list.append(rz_per_trot)
		trot_calls_list.append(trot_calls)
		t_count_list.append(t_count)
		cost_arr.append(runtime)

	if rank == 0:
		print(f'\n\n\nFinished all spectral calculations after a total of {TIMES[-1] - TIMES[time_index_before_spectra]:.0f} seconds, starting plots...')
		with open(mol_name + '.txt', 'a') as f:
			sys.stdout = f
			print(f'argv = {sys.argv[1:]}')
			print(f"Number of modals per mode is {num_modals}")
			print(f'dip_anh={dip_anh}')
			print(f'is_loc={is_loc}')
			print(f'num_couplings={num_couplings}')
			print(f'epsilon_in_cm={epsilon_in_cm}')
			print(f'trotter_order={trotter_order}')
			print(f'r_arr={r_arr}, kmax_arr={k_arr}, shots_arr={shots_arr}')
			#print(f'keff_arr={keff_list}')
			if rpt is not None:
				print(f'rpt={rpt}')
			print(f'R_z gates per call to trotter oracle: {np.asarray(rz_per_trot_list)}')
			print(f'T gates per R_z gate:{np.asarray(rz_synth_num)}')
			print(f'Number of calls to trotter oracle: {np.asarray(trot_calls_list)}')
			print(f'T-gate depth: {np.asarray(t_count_list)}')
			print(f'Total number of fragments: {tot_frags}')
			print(f'Total run-time in hours for clock-rate of 1e6 Hz: {np.asarray(cost_arr)}')
			print(f'Total time for running workflow was {time() - TIMES[0]:.0f} seconds')
			print('\n##############################')
			print('##############################\n')
			print('  ')

		sys.stdout = sys.__stdout__
