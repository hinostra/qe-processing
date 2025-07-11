import os
import re
import glob
import numpy as np

def WALL_time_to_seconds(s):
    """
    Transform string of CPU/WALL time in output files of QE
    to time in seconds
    """
    
    # Remove extra spaces
    s = s.strip().replace(" ", "")
    # Extract hours, minutes, seconds using regex
    hours = re.search(r"(\d+(?:\.\d+)?)h", s)
    minutes = re.search(r"(\d+(?:\.\d+)?)m", s)
    seconds = re.search(r"(\d+(?:\.\d+)?)s", s)

    h = float(hours.group(1)) * 3600 if hours else 0
    m = float(minutes.group(1)) * 60 if minutes else 0
    s = float(seconds.group(1)) if seconds else 0

    return h + m + s

def read_ecut(directory):
    """
    Analysis of convergence with varying cutoff energy.
    Read output file of QE.
    Args:
    - directory: directory with input and output files

    Output:
    - cutoff energies (np.array)
    - scf total energies (np.array)
    - cpu-time in seconds (np.array)
    """

    output_files = [x for x in os.listdir(directory) if x.endswith('out')]
    cutoff_energy = []
    scf_energy = []
    times = []
    for filename in output_files:
        filepath = os.path.join(directory, filename)

        with open(filepath, 'r') as f:
            for line in f.readlines():
                if line.startswith('     Number of MPI processes:'):
                    nmpis = int(line.split(':')[1].split(' ')[-1])
                if line.startswith('     kinetic-energy cutoff'):
                    cutoff = float(line.split('=')[1].split(' ')[-3])
                    cutoff_energy.append(cutoff)
                if line.startswith('!    total energy'):
                    total = float(line.split('=')[1].split(' ')[-2])
                    scf_energy.append(total)
                if line.startswith('     PWSCF        :'):
                    time = WALL_time_to_seconds(line.split('CPU')[1].split('WALL')[0])
                    time *= nmpis
                    times.append(time)

    cutoff_energy = np.array(cutoff_energy)
    scf_energy = np.array(scf_energy)
    times = np.array(times)
    idxs = np.argsort(cutoff_energy)

    return cutoff_energy[idxs], scf_energy[idxs], times[idxs]

def read_kpoints(directory):
    """
    Analysis of convergence with varying number of kpoints.
    Read input and output file of QE.
    Args:
    - directory: directory with input and output files

    Output:
    - kpoints (k1, k2, k3) (2D np.array)
    - scf total energies (np.array)
    - cpu-time in seconds (np.array)
    """

    output_files = [x for x in os.listdir(directory) if x.endswith('out')]
    input_files = [x[:-3]+'in' for x in output_files]

    kpoints =[]
    scf_energy = []
    times = []

    for ifile, ofile in zip(input_files, output_files):
        ifilepath = os.path.join(directory, ifile)

        with open(ifilepath, 'r') as f:
            read = False
            for line in f.readlines():
                if line.startswith('K_POINTS'): 
                    read = True
                    continue
                if line.startswith('CELL_PARAMETERS'):
                    break
                if read:
                    kp = line.split(' ')
                    kp = [k for k in kp if len(kp)>0]
                    kp = kp[:3]
                    kp = [int(k) for k in kp]
                    kpoints.append(kp)

        ofilepath = os.path.join(directory, ofile)
        with open(ofilepath, 'r') as f:
            for line in f.readlines():
                if line.startswith('     Number of MPI processes:'):
                    nmpis = int(line.split(':')[1].split(' ')[-1])
                if line.startswith('!    total energy'):
                    total = float(line.split('=')[1].split(' ')[-2])
                    scf_energy.append(total)
                if line.startswith('     PWSCF        :'):
                    time = WALL_time_to_seconds(line.split('CPU')[1].split('WALL')[0])
                    time *= nmpis
                    times.append(time)

    kpoints = np.array(kpoints)
    scf_energy = np.array(scf_energy)
    times = np.array(times)
    idxs = np.argsort(kpoints[:,0])

    return kpoints[idxs], scf_energy[idxs], times[idxs]

def read_input_rlx(input_file):
    """
    Read input file of QE for geometric optimization
    to extract initial cell parameters, atomic coordinates,
    and atom labels.
    Args:
    - input_file: path to input file (vc-relax)

    Output:
    - cell parameters or lattice vectors (2D np.array)
    - xyz coordinates of atoms (2D np.array)
    - atomic labels (list)
    """

    # read input file
    inlines = open(input_file, 'r').readlines()
    extract = False
    selected_lines = []
    for line in inlines:
        # extract relevant information
        if extract: 
            tokens = []
            for x in line.split(' '):
                if x.endswith('\n'): x = x[:-2] # avoid linejumps
                if len(x)==0: continue  # avoid blank spaces
                tokens.append(x)
            selected_lines.append(tokens)

        if line.startswith('ATOMIC_POSITIONS'):
            extract = True
        elif line.startswith('K_POINTS'):
            extract = False
        elif line.startswith('CELL_PARAMETERS'):
            extract = True

    # initial cell vectors
    cell_parameters = np.array([[float(x) for x in y] for y in selected_lines[-3:]])

    # atomic labels
    _labels = [x[0] for x in selected_lines[:-4]]   # chemical symbol
    labels = []
    # label = chemical symbol + atom index
    for i in range(len(_labels)):
        labels.append(_labels[i] + str(_labels[:i+1].count(_labels[i])))

    # initial coordinates
    xyz = np.array([[float(x) for x in y[1:]] for y in selected_lines[:-4]])
    xyz = np.matmul(xyz, cell_parameters)
    return cell_parameters, xyz, labels

def read_output_rlx(output_file):
    """
    Read output file of QE for geometric optimization
    to extract final cell parameters, atomic coordinates,
    and atom labels.
    Args:
    - output_file: path to output file (vc-relax)

    Output:
    - cell parameters or lattice vectors (2D np.array)
    - xyz coordinates of atoms (2D np.array)
    - atomic labels (list)
    """

    # read output file
    inlines = open(output_file, 'r').readlines()
    extract = False
    outlines = []
    for line in inlines:
        # extract final coordinates
        if extract: outlines.append(line)

        if line.startswith('Begin final coordinates'):
            extract = True
        elif line.startswith('End final coordinates'):
            extract = False

    selected_lines = []
    for line in outlines[:-1]:
        # extract relevant information
        if extract: 
            tokens = []
            for x in line.split(' '):
                if x.endswith('\n'): x = x[:-2] # avoid linejumps
                if len(x)==0: continue  # avoid blank spaces
                tokens.append(x)
            selected_lines.append(tokens)

        if line.startswith('CELL_PARAMETERS'):
            extract = True

    # final cell vectors
    cell_parameters = np.array([[float(x) for x in y] for y in selected_lines[:3]])

    # atomic labels
    _labels = [x[0] for x in selected_lines[5:]]   # chemical symbol
    labels = []
    # label = chemical symbol + atom index
    for i in range(len(_labels)):
        labels.append(_labels[i] + str(_labels[:i+1].count(_labels[i])))

    # final coordinates
    xyz = np.array([[float(x) for x in y[1:]] for y in selected_lines[5:]])
    xyz = np.matmul(xyz, cell_parameters)
    
    return cell_parameters, xyz, labels

def read_energies_forces_rlx(output_file):
    """
    Read output file of QE for geometric optimization
    to extract cell parameters, energies, and forces
    for each iteration
    Args:
    - output_file: path to output file (vc-relax)

    Output:
    - cell parameters or lattice vectors for each iteration (3D np.array)
    - energies (np.array)
    - forces (np.array)
    """

    # read output file
    inlines = open(output_file, 'r').readlines()
    extract_coor = False
    outlines = []
    energies = []
    forces = []
    for line in inlines:
        if line.startswith('CELL_PARAMETERS (angstrom)'):
            extract_coor = True
            continue
        elif line.startswith('ATOMIC_POSITIONS (crystal)'):
            extract_coor = False
            continue
        elif line.startswith('!    total energy'):
            tokens = [x for x in line.split(' ') if len(x)!=0]
            energies.append(float(tokens[-2]))
        elif line.startswith('     Total force'):
            tokens = [x for x in line.split(' ') if len(x)!=0]
            forces.append(float(tokens[3]))

        # extract final coordinates
        if extract_coor: 
            if line.endswith('\n'): line = line[:-2]
            if len(line)==0: continue
            outlines.append(line)

    cell_parameters = [[float(x) for x in y.split(' ') if len(x)!=0] for y in outlines]
    cell_parameters = [[cell_parameters[i*3], cell_parameters[i*3+1], cell_parameters[i*3+2]] for i in range(len(cell_parameters)//3)]
    cell_parameters = np.array(cell_parameters)

    energies = np.array(energies)
    forces = np.array(forces)
    
    return cell_parameters, energies, forces

def read_output_scf(output_file):
    """
    Read output of scf energy calculation
    to extract energy contributions to
    total energy.
    Args:
    - output_file: path to output file

    Output:
    - (h0Ene, hartreeEne, xcEne, ewaldEne, pawEne) in a np.array
    """

    outlines = open(output_file, 'r').readlines()

    select = False
    eTerms = []
    contributions = ['!', 'one-electron', 'hartree', 'xc', 'ewald', 'one-center']
    for line in outlines:
        if line.startswith('!'): select = True
        if line.startswith('     convergence has been achieved'): select = False
        if not select: continue
        tokens = [x for x in line.split(' ') if len(x)>0]
        if tokens[0] not in contributions: continue
        eTerms.append(float(tokens[-2]))

    total = eTerms[0]
    h0Ene, hartreeEne, xcEne, ewaldEne, pawEne = eTerms[1:]

    print('One-Electron:', h0Ene)
    print('   + Hartree:', h0Ene+hartreeEne)
    print('        + XC:', h0Ene+hartreeEne+xcEne)
    print('     + Ewald:', h0Ene+hartreeEne+xcEne+ewaldEne)
    print('       + PAW:', h0Ene+hartreeEne+xcEne+ewaldEne+pawEne)
    print('Total Energy: ', total, 'Ry')
    return eTerms

def read_dos(directory, elements=['Y','Al','O']):
    """
    Read dat output files of a Density of States (DOS)
    calculation to extract total DOS, projected DOS
    per chemical specie, and per orbital.
    Args:
    - directory: directory with output files

    Output:
    - directory with arrays of: energy, dos, pdos_{el}, pdos_{el}_{or}
    """

    output = {}

    # ldos total
    ldosfile = [x for x in os.listdir(directory) if x.endswith('pdos_tot')]
    assert len(ldosfile)==1, f"xxxxx.pdos_tot should be in {directory}"
    ldosfile = ldosfile[0]
    data = np.loadtxt(os.path.join(directory, ldosfile))
    energy, ldos_tot = data[:, 0], data[:, 1]
    output['energy'] = energy
    output['dos'] = ldos_tot

    # ldos per element
    for el in elements:
        # get files that correspond to element's pdos
        file_pattern = f'*.dat.pdos_atm#*({el})*'
        files_with_el = glob.glob(directory + '/' + file_pattern)
        pdosfiles = [x for x in files_with_el]

        # extract ldos from files
        ldos = []
        for fname in pdosfiles:
            data = np.loadtxt(fname)
            ldos.append(data[:,1])
        nmax = np.max([x.size for x in ldos])
        ldos = [np.concatenate([x, np.zeros(nmax-x.size)]) for x in ldos]
        ldos = np.array(ldos)

        # extract specific orbital dos
        for i, fname in enumerate(pdosfiles):
            orb = fname.split('(')[-1][0]
            if f"pdos_{el}_{orb}" not in output: 
                output[f"pdos_{el}_{orb}"] = np.zeros(nmax)
            output[f"pdos_{el}_{orb}"] += ldos[i, :]
        # dos by element
        ldos = np.sum(ldos, axis=0)
        output[f"pdos_{el}"] = ldos

    return output

def read_output_bands(output_pp, output_bands):
    """
    Read output files of processing band calculations
    to extract valence and conduction bands.
    Args:
    - output_pp: output file of &BANDS/bands.x to extract
                 high-symmetry points (printed information)
    - output_bands: output dat file with bands 
                 likely named XXXXX.dat.gnu
    
    Output:
    - directory with arrays of: kpoints, valence_band, 
      conduction_band, Fermi_energy, high_symmetry_points
    """

    # get high symmetry points
    hs_points = []
    with open(output_pp, 'r') as f:
        for line in f.readlines():
            if line.startswith('     high-symmetry point:'):
                hs_points.append(float(line.split(' ')[-1]))

    # extract bands
    data = np.loadtxt(output_bands)
    k = np.unique(data[:,0])
    _bands = np.reshape(data[:, 1], (-1, len(k)))

    # average energy in a energy level
    ll_band = _bands.mean(axis=1)
    # check for gaps larger than 1eV
    mask = ll_band[1:]-ll_band[:-1] > 1
    # midpoint of energy gap
    gap = (ll_band[1:][mask] + ll_band[:-1][mask])/2
    # define valence band
    valence_band = _bands[(ll_band>gap[-2])&(ll_band<gap[-1]),:]
    # define conduction band
    conduction_band = _bands[(ll_band>gap[-1]),:]
    # get Fermi energy
    eFermi = valence_band.max()

    # print information
    print(f'Fermi energy = {eFermi:.4f} eV')

    print(f'Valence band = [{valence_band.min():.4f},{valence_band.max(): .4f}]')
    idxs = np.where(valence_band==valence_band.max())
    ii, jj = idxs[0], idxs[1]
    for n in range(len(ii)):
        i, j = ii[n], jj[n]
        top_valence = valence_band[i, j]
        print(f'Band {i} k={k[j]:.4f} with {top_valence:.4f} eV')

    print(f'Conduction band = [{conduction_band.min(): .4f}, {conduction_band.max():.4f}]')
    idxs = np.where(conduction_band==conduction_band.min())
    ii, jj = idxs[0], idxs[1]
    for n in range(len(ii)):
        i, j = ii[n], jj[n]
        bottom_conduction =  conduction_band[i, j]
        print(f'Band {i} k={k[j]:.4f} with {bottom_conduction:.4f} eV')

    print(f'Energy gap = {conduction_band.min()-valence_band.max(): .4f} eV')

    output = {'kpoints': k,
              'valence_band': valence_band-eFermi,
              'conduction_band': conduction_band-eFermi,
              'Fermi_energy': eFermi,
              'high_symmetry_points': np.array(hs_points)}

    return output