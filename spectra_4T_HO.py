import h5py
import numpy as np
import sys
from vibrant.vscf import VSCF_Fock
from vibrant.algorithm.lt_qpe import dipole_based_normalizer
from vibrant.qubit_utils import vib_from_modes
from vibrant.system_prep.initial_state import dipole_to_vector, vscf_ground_state_second_quantized, vib_occupation_to_second_quantized
from vibrant.algorithm.simulate import post_process, run_mpi_generic
from vibrant.algorithm.trotter.cform_error import r_from_pt, get_grouped_ham, get_CF_ham 
from time import time
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import itertools
import pickle

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()

'''
INPUTS FOR SIMULATOR:
python run_LT.py mol_name num_modals r_arr (k_arr) ((shots_arr))
arguments in parenthesis are optional

e.g.
mol_name = H2S
num_modals = 2

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
is_loc = input-loc #whether Hamiltonian comes from localized modals
num_couplings = input-num-coup #number of mode couplings for the Hamiltonian
epsilon_in_cm = 7 #eigenvalue accuracy for PT bound
trotter_order = 2 #order of Trotter expansion
frag_method = 'input-frag-method'
frag_tol = input-frag-tol
pauli_tol = input-pauli-tol 
tau  = 250  # (in a.u.) this is one time-step. We have total of kmax such steps for the full propagation.
# tau will be divided into r smaller steps of tau/r for Trotter propagation.

dark_plots = False
max_width = 3 #maximum width of line for plot
min_width = 1 #minimum width of line for plot

au_to_cm = 219475
eta = 10


mol_name = sys.argv[1]
num_modals = int(sys.argv[2])

if rank == 0:
	print(f"Molecule name is {mol_name}")
	print(f"Number of modals per mode is {num_modals}")
	print(f"Running with PennyLane device {pl_device}")
	print(f"Mode localization is set to {is_loc}")
	print(f"Number of mode couplings included is {num_couplings}")
	print(f"Anharmonic dipole components inclusion is set to {dip_anh}")


r_arr_string = sys.argv[3].split(',')
num_rs = len(r_arr_string)
r_arr = []
for r_string in r_arr_string:
	if r_string == '_':
		r_arr.append(None)
	else:
		r_arr.append(int(r_string))
if rank == 0:
	print(f"Using array of r's for simulations {r_arr}")

if len(sys.argv) > 4:
	k_arr_string = sys.argv[4].split(',')
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

if len(sys.argv) > 5:
	shots_arr_string = sys.argv[5].split(',')
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
	H1 = H1_full[:,:num_modals,:num_modals]
	if num_couplings > 1:
		H2_full = f['T2'][()]
		H2 = H2_full[:,:,:num_modals,:num_modals,:num_modals,:num_modals]
	if num_couplings > 2:
		H3_full = f['T3'][()]
		H3 = H3_full[:,:,:,:num_modals,:num_modals,:num_modals,:num_modals,:num_modals,:num_modals]

	D1_full = f['taylor_D1'][()]
	D1 = D1_full[:,:,:num_modals,:num_modals]
	if dip_anh:
		D2_full = f['taylor_D2'][()]
		D2 = D2_full[:,:,:,:num_modals,:num_modals,:num_modals,:num_modals]
	else:
		D2_full = None
	f.close()

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
#fock_object.VSCF(verbose=False)

def dip_lists(D1, D2, modals):
    dip_vect_x, dip_norm_x = dipole_to_vector(D1,modals=modals,orientation='x', dip2=D2)
    dip_vect_y, dip_norm_y = dipole_to_vector(D1,modals=modals,orientation='y', dip2=D2)
    dip_vect_z, dip_norm_z = dipole_to_vector(D1,modals=modals,orientation='z', dip2=D2)
    wf_list = [dip_vect_x, dip_vect_y, dip_vect_z]
    dip_norm_list = [dip_norm_x, dip_norm_y, dip_norm_z]

    return wf_list, dip_norm_list

#vscf_energy = fock_object.fock_energy()


modals = fock_object.modals
#Hnorm = dipole_based_normalizer(fock_object, dip_arr, modals, verbose=False)
#freq_max = 0.99 * 4 * Hnorm * au_to_cm
#Don't need to calculate Hnorm. It will be Hnorm = 2*pi/tau. The variable Hnorm is actually Hnorm/4 = pi/2/tau
Hnorm = np.pi / 2 / tau
freq_max = 0.99 * 4 * Hnorm * au_to_cm  # 4 * Hnorm is the actual Hnorm


if rank == 0:
	print(f"Norm of Hamiltonian is {Hnorm:.2e}, corresponding to spectral window up to {freq_max:.0f} cm-1")
TIMES.append(time())
if rank == 0:
	print(f"Time elapsed for doing VSCF and finding norm was {TIMES[-1]-TIMES[-2]:.2f} seconds")

rpt = None
if None in r_arr:
	if frag_method in ['Pauli', 'FC', 'QWC']:
		rpt = r_from_pt(fock_object, dip_arr, modals, Hnorm, frag_method = frag_method, vscf_flag = False, epsilon_in_cm=epsilon_in_cm, verbose=True)
	elif frag_method == 'CF':
		rpt = r_from_pt(fock_object, dip_arr, modals, Hnorm, frag_method = frag_method, vscf_flag = False, epsilon_in_cm=epsilon_in_cm, verbose=True, mol_name = mol_name, is_loc = is_loc, num_couplings = num_couplings, frag_tol = frag_tol)

	if rank == 0:
		print(f"Found r={rpt} using perturbation theory, bringing to integer {np.ceil(rpt)}")
	rpt = int(np.ceil(rpt))
rpt = COMM.bcast(rpt, root=0)

for num_sim in range(num_rs):
	if r_arr[num_sim] is None:
		r_arr[num_sim] = rpt

TIMES.append(time())
if rank == 0:
	print(f"Time elapsed for obtaining perturbative r was {TIMES[-1]-TIMES[-2]:.2f} seconds")

au_max = freq_max / au_to_cm
freqs = np.linspace(0,au_max,10000)

#Hvscf, Dvscf = fock_object.vscf_objects(dip_arr, modals)
num_qubits, pl_ham = vib_from_modes(ham_arr, modals, ps=True, cutoff = pauli_tol)

#Create indentity matrix. Used to create id * |trial_state>.
id_mat = np.zeros(np.shape(dip_arr[0]))
for rho in range(3):
	for i in range(len(modals)):
		id_mat[rho,i,:,:] = np.eye(num_modals)

init_wf, _ = dipole_to_vector(id_mat,modals=modals)

#Expecatiion value of the trial state with Hamiltonian.
Eexp = np.vdot(init_wf, pl_ham.to_mat() @ init_wf) 

#if dip_anh == False:
#	D2_vscf = None
#else:
#	D2_vscf = Dvscf[1]

TIMES.append(time())
if rank == 0:
	print(f"Time elapsed for building VSCF matrices and PL operators was {TIMES[-1]-TIMES[-2]:.2f} seconds")
time_index_before_spectra = len(TIMES) - 1

if rank == 0:
	print('\n##############################')
	print("Starting spectrum calculations")
	print('##############################\n')
#wf_list, dip_norm_list = dip_lists(Dvscf[0], D2_vscf, modals)
wf_list, dip_norm_list = dip_lists(dip_arr[0], dip_arr[1], modals)

if trotter_order == 1:
	method = 'trotter'
elif trotter_order > 1:
	method = f'trotter{trotter_order}'
if rank == 0:
	print(f'Trotter simulations will be ran with method {method}')

keff_list = []
spectra_list = []
for sim_num in range(num_rs):
	my_r = r_arr[sim_num]
	if my_r is None:
		my_r = rpt
	my_kmax = k_arr[sim_num]
	my_shots = shots_arr[sim_num]
	if rank == 0:
		print(f'\n\nStarting simulation for r={my_r}')
		if my_r == 0:
			print(f'r=0, implementing exact time evolution through diagonalization!')
	my_name = data_folder + f"{mol_name}_M{num_modals}_r{my_r}"
	
	if my_r == 0:
		my_expectations, my_L_k = run_mpi_generic(pl_ham, Hnorm, wf_list, dip_norm_list, num_steps_per_k=my_r, device_type=pl_device, parallel_cartesian=True, prefix=my_name, method_name=method, eta=eta, kmax = my_kmax)
	else:
		if frag_method == 'Pauli' :
			ham = pl_ham.hamiltonian()
		elif frag_method == 'FC' or frag_method == 'QWC' :
			ham = get_grouped_ham(pl_ham.hamiltonian(), frag_method)
		elif frag_method == 'CF':
			_, ham = get_CF_ham(modals, mol_name = mol_name, is_loc = is_loc, num_couplings = num_couplings, frag_tol = frag_tol, vscf_flag = False)
		else:
			raise ValueError(f"Fragmentation method is none of the allowed: Pauli, FC, QWC or CF")

		my_expectations, my_L_k = run_mpi_generic(ham, Hnorm, wf_list, dip_norm_list, num_steps_per_k=my_r, device_type=pl_device, parallel_cartesian=True, prefix=my_name, method_name=method, eta=eta, kmax = my_kmax)

	if rank == 0:
		print(f'Got quantum simulation results, simulating for kmax={my_kmax} and shots={my_shots}')
	my_spectrum, my_keff = post_process(my_expectations, Hnorm, my_L_k, dip_norm_list, shots=my_shots, kmax=my_kmax, eta=eta)

	TIMES.append(time())
	if rank == 0:
		print(f"Time elapsed for simulation {sim_num+1} out of {num_rs} was {TIMES[-1]-TIMES[-2]:.2f} seconds")
	#spectra_list.append([my_spectrum(w + vscf_energy) for w in freqs])
	spectra_list.append([my_spectrum(w + Eexp) for w in freqs])

	keff_list.append(my_keff)


widths = np.linspace(max_width, stop=min_width, num=num_rs)
if rank == 0:
	print(f'\n\n\nFinished all spectral calculations after a total of {TIMES[-1] - TIMES[time_index_before_spectra]:.0f} seconds, starting plots...')

plt.figure(figsize=(10,6))
for sim_num in range(num_rs):
	if r_arr[sim_num] == 0:
		my_label = 'exact'
	else:
		my_label = f'r={r_arr[sim_num]}'
	if has_k:
		my_label = my_label + f', keff={keff_list[sim_num]}'
	if has_shots:
		my_label = my_label + f', shots={shots_arr[sim_num]}'
	plt.plot(freqs * au_to_cm, spectra_list[sim_num], label=my_label, linewidth=widths[sim_num])

if is_loc:
    with open(f'{mol_name}_{num_modals}M_{frag_method}_4T2M_loc.out','wb') as f:
        pickle.dump([freqs* au_to_cm, spectra_list], f)
else:
    with open(f'{mol_name}_{num_modals}M_{frag_method}_4T{num_couplings}M.out','wb') as f:
        pickle.dump([freqs* au_to_cm, spectra_list], f)


plt.xlabel('Excitation energy (cm-1)')
plt.ylabel('Absorption intensity (arb. units)')
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
pretty_name = mol_name.translate(SUB)
plt.title(f"IR absorption spectrum for {pretty_name}")
plt.legend()

def unique_file(basename, ext):
    actualname = "%s" % (basename)
    c = itertools.count()
    while os.path.exists(actualname + f'.{ext}'):
        actualname = "%s_(%d)" % (basename, next(c))
    return actualname

fig_name = unique_file(data_folder + mol_name, 'png')
if rank == 0:
	plt.savefig(fig_name + '.png')
	print(f"Saved figure as {fig_name + '.png'}")
	print(f'r_arr={r_arr}, kmax_arr={k_arr}, shots_arr={shots_arr}')
	print(f'keff_arr={keff_list}')
	if rpt is not None:
		print(f'rpt={rpt}')
	print(f'argv = {sys.argv[1:]}')
	with open(fig_name + '.txt', 'w') as f:
		sys.stdout = f
		print(f"Saved figure as {fig_name + '.png'}")
		print(f'r_arr={r_arr}, kmax_arr={k_arr}, shots_arr={shots_arr}')
		print(f'keff_arr={keff_list}')
		if rpt is not None:
			print(f'rpt={rpt}')
		print(f'argv = {sys.argv[1:]}')
		print(f'dip_anh={dip_anh}')
		print(f'is_loc={is_loc}')
		print(f'num_couplings={num_couplings}')
		print(f'epsilon_in_cm={epsilon_in_cm}')
		print(f'trotter_order={trotter_order}')
		print(f'Total time for running workflow was {time() - TIMES[0]:.0f} seconds')
	sys.stdout = sys.__stdout__
