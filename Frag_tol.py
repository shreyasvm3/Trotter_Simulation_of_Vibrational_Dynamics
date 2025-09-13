import h5py
import pickle
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from vibrant.qubit_utils import vib_from_modes
from vibrant.qubit_utils.select_physical import fold_mat

au_to_cm = 219474.63

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

def load_frags(mol, nmodals, is_loc = True, num_couplings = 2):
	'Read in fragments from file'

	if is_loc: 
		with open(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/VSCF_fragments/{mol}_VSCFfragments_{str(nmodals)}M_{str(num_couplings)}MC_loc.out', 'rb') as f:
			params, norms = pickle.load(f)
	else:
		with open(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/VSCF_fragments/{mol}_VSCFfragments_{str(nmodals)}M_{str(num_couplings)}MC.out', 'rb') as f:
			params, norms = pickle.load(f)

	return params, norms

def H_chrs_frag(mol, nmodes, nmodals, is_loc = True, num_couplings = 2):

	global au_to_cm

	if is_loc:
		f = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/VSCF_tensors/{mol}_VSCF_{nmodals}M_{num_couplings}MC_loc.hdf5', 'r')
	else:
		f = h5py.File(rf'/home/i/izmaylov/smalpath/Christiansen_Fragments/4T3M/VSCF_tensors/{mol}_VSCF_{nmodals}M_{num_couplings}MC.hdf5', 'r')
	
 
	H1_og = f['H1'][()] #one-mode Hamiltonian in VSCF  basis
	params, _ = load_frags(mol, nmodals, is_loc = is_loc, num_couplings = num_couplings)
	nf = len(params)

	if num_couplings == 2:
		from vibrant.algorithm.trotter.GFRO_utils_2MC import get_fragment

		H2_frag = np.zeros((nf,nmodes,nmodes,nmodals,nmodals,nmodals,nmodals))
		for i in range(nf):
			H2_frag[i] = get_fragment(params[i], nmodals, nmodes)
		for i in range(1,nf):
			H2_frag[i] += H2_frag[i-1]
		H_arr = [H1_og * au_to_cm, H2_frag]

	if num_couplings == 3:
		from vibrant.algorithm.trotter.GFRO_utils_3MC import get_fragment

		H2_frag = np.zeros((nf,nmodes,nmodes,nmodals,nmodals,nmodals,nmodals))
		H3_frag = np.zeros((nf,nmodes,nmodes,nmodes,nmodals,nmodals,nmodals,nmodals,nmodals,nmodals))
		for i in range(nf):
			H2_frag[i], H3_frag[i] = get_fragment(params[i], nmodals, nmodes)
		for i in range(1,nf):
			H2_frag[i] += H2_frag[i-1]
			H3_frag[i] += H3_frag[i-1]            

		H_arr = [H1_og * au_to_cm, H2_frag, H3_frag]

	return H_arr

def save_frag_eigs(mol, nm, ev, is_loc = True, num_couplings = 2):
    
    if is_loc:
        filename = mol + '_VSCF_'+ str(nm) + 'M_2MC_loc_evs.out'
    else :
        filename = mol + '_VSCF_'+ str(nm) + 'M_'+ str(num_couplings) + 'MC_evs.out'

    with open(filename, 'wb') as f:
        pickle.dump(ev, f)
    return None

def save_exact_eigs(mol, nm, ev, is_loc = True, num_couplings = 2):

    if is_loc:
        filename = mol + '_VSCF_'+ str(nm) + 'M_2MC_loc_exact_evs.out'
    else :
        filename = mol + '_VSCF_'+ str(nm) + 'M_'+ str(num_couplings) + 'MC_exact_evs.out'

    with open(filename, 'wb') as f:
        pickle.dump(ev, f)
    return None

mol_list    = ['H2S', 'CO2', 'CH2O', 'C2N2']
modals_min  = [2,2,2,2]
modals_max  = [4,3,2,2]
is_loc = input-loc #whether Hamiltonian comes from localized modals
num_couplings = input-num-coup #number of mode couplings for the Hamiltonian
ev0      = []
ev1      = []
ev2      = []
i = 0
for mol in mol_list:
    print(f'Starting molecule:{mol}')
    for nm in range(modals_min[i],modals_max[i]+1):
        ex_evs = []
        H_arr, _ = read_vscf_tensors(mol, nm, is_loc = is_loc, num_couplings = num_couplings)
        nmodes = H_arr[0].shape[0]
        #obtain operators in VSCF basis
        modals = [nm] * nmodes
        print(f'Starting modals:{modals}')
        _, ham = vib_from_modes(H_arr, modals=modals, ps = True) #build Hamiltonian operator
        Hmat = ham.to_mat()
        Hred = fold_mat(Hmat, modals)
        ex_evals = np.linalg.eigvalsh(Hred) * au_to_cm
        ex_evs.append(ex_evals)
        save_exact_eigs(mol, nm, ex_evs, is_loc = is_loc, num_couplings = num_couplings)
        print(f'Exact eigenvalues: {ex_evals[:4]}')
        Hfrag = H_chrs_frag(mol, nmodes, nm, is_loc = is_loc, num_couplings = num_couplings)
        n_frags = Hfrag[1].shape[0]
        print(f'Total number of 2/3-mode fragments:{n_frags}')
        evs = []       
        ev0 = []
        ev1 = []
        ev2 = [] 
        ev3 = []
        ev4 = [] 
        H_arr = [Hfrag[0]]
        _, ham = vib_from_modes(H_arr, modals=modals, ps = True) #build Hamiltonian operator
        Hmat = ham.to_mat()
        Hred = fold_mat(Hmat, modals)
        evals = np.linalg.eigvalsh(Hred)
        evs.append(evals)
        print(f'Number of fragments: 0, Evs:, {evals[:4]}')
        save_frag_eigs(mol, nm, evs, is_loc = is_loc, num_couplings = num_couplings)
        ev0.append(evals[0]-ex_evals[0])
        ev1.append(evals[1]-ex_evals[1])
        ev2.append(evals[2]-ex_evals[2])
        ev3.append(evals[3]-ex_evals[3])
        ev4.append(evals[4]-ex_evals[4])
        for nf in range(n_frags):
            if num_couplings == 2:
                H_arr = [Hfrag[0], Hfrag[1][nf]]
            elif num_couplings == 3:
                H_arr = [Hfrag[0], Hfrag[1][nf], Hfrag[2][nf]]
               
            _, ham = vib_from_modes(H_arr, modals=modals, ps = True) #build Hamiltonian operator
            Hmat = ham.to_mat()
            Hred = fold_mat(Hmat, modals)
            evals = np.linalg.eigvalsh(Hred)
            evs.append(evals)
            print(f'Number of fragments: {nf+1}, Evs:, {evals[:4]}')#,file=fout)
            save_frag_eigs(mol, nm, evs, is_loc = is_loc, num_couplings = num_couplings)
            ev0.append(evals[0]-ex_evals[0])
            ev1.append(evals[1]-ex_evals[1])
            ev2.append(evals[2]-ex_evals[2])
            ev3.append(evals[3]-ex_evals[3])
            ev4.append(evals[4]-ex_evals[4])
        fig, ax = plt.subplots(constrained_layout=True)
        params, norms = load_frags(mol, nm, is_loc = is_loc, num_couplings = num_couplings)
        x = norms
        ax.loglog(norms,np.abs(ev0),'-')
        ax.loglog(norms,np.abs(ev1),'-',)
        ax.loglog(norms,np.abs(ev2),'-',)
        ax.loglog(norms,np.abs(ev3),'-',)
        ax.loglog(norms,np.abs(ev4),'-',)

        ax.set_xlabel("Norm")
        ax.set_ylabel("Error in eigenvalues (cm-1)")
        ax.legend(["ev0","ev1","ev2","ev3","ev4"],loc='lower right')
        if is_loc:
            ax.set_title(f'{mol} {str(nm)}M 2MC Loc')
            plt.savefig(f'{mol}_VSCF_frag_tol_{nm}M_2MC_loc.png')
        else:
            ax.set_title(f'{mol} {str(nm)}M {num_couplings}MC')
            plt.savefig(f'{mol}_VSCF_frag_tol_{nm}M_{num_couplings}MC.png')

        ax.grid()
        #plt.show()                
    i += 1
