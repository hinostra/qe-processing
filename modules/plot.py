import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_etotal_cutoff(cutoff_energy, scf_energy, ax=None, **kwargs):
    """
    Plot convergence of total energy with varying cutoff energy.
    Args:
    - cutoff_energy: array with values of cut off energies
    - scf_energy: array with values of total energies
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure
    
    ax.plot(cutoff_energy, scf_energy, **kwargs)
    ax.set_xlabel('Energía de corte [Ry]')
    ax.set_ylabel('Energía total [Ry]')

def plot_ediff_cutoff(cutoff_energy, scf_energy, threshold=1e-4, ax=None, **kwargs):
    """
    Plot difference of total energy of consecutive cutoff energies.
    Args:
    - cutoff_energy: array with values of cut off energies
    - scf_energy: array with values of total energies
    - threshold=1e-4: horizontal line of upper limit
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure
    
    energy_diff = np.abs(scf_energy[1:] - scf_energy[:-1])
    ax.plot(cutoff_energy[1:], energy_diff, **kwargs)
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], 
              linestyle='--', color='#9e9e9e')
    ax.set_xlabel('Energía de corte [Ry]')
    ax.set_ylabel(r'$\Delta E_\text{cut}$ [Ry]')
    ax.set_yscale('log')

def plot_times(x, times, ax=None, **kwargs):
    """
    Plot cpu-times
    Args:
    - x: array with values of parameter
    - times: array with cpu-times in seconds
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure
    
    # time in hours
    ax.plot(x, times/3600, **kwargs)
    ax.set_ylabel('CPU-time [CPU-horas]')

def plot_etotal_kpoints(kpoints, scf_energy, ax=None, **kwargs):
    """
    Plot convergence of total energy with varying number of kpoints.
    Args:
    - kpoints: array with values of kpoints
    - scf_energy: array with values of total energies
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure

    x = np.arange(kpoints.shape[0])
    ax.plot(x, scf_energy, **kwargs)
    ax.set_xticks(x, [f"({int(i)},{int(j)},{int(k)})" for (i,j,k) in kpoints])
    ax.tick_params(axis='x', labelrotation=30)
    ax.set_xlabel(r'$(k_1,k_2,k_3)$')
    ax.set_ylabel('Energía total [Ry]')

def plot_ediff_kpoints(kpoints, scf_energy, threshold=1e-4, ax=None, **kwargs):
    """
    Plot difference of total energy of consecutive kpoints.
    Args:
    - kpoints: array with values of kpoints
    - scf_energy: array with values of total energies
    - threshold=1e-4: horizontal line of upper limit
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure
    
    x = np.arange(kpoints.shape[0])
    energy_diff = np.abs(scf_energy[1:] - scf_energy[:-1])
    ax.plot(x[1:], energy_diff, **kwargs)
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], 
              linestyle='--', color='#9e9e9e')
    ax.set_xticks(x, [f"({int(i)},{int(j)},{int(k)})" for (i,j,k) in kpoints])
    ax.tick_params(axis='x', labelrotation=30)
    ax.set_xlabel(r'$(k_1,k_2,k_3)$')
    ax.set_ylabel(r'$\Delta E_\text{cut}$ [Ry]')
    ax.set_yscale('log')

def plot_energy_rlx(energies, ax=None, **kwargs):
    """
    Plot difference of the energy calculated at each iteration
    and the minimum energy obtained at the end of the optimization.
    Args:
    - energies: array of energies
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure

    x = np.arange(len(energies))
    y = energies - energies.min()
    ax.plot(x, y*1e3, **kwargs)
    ax.set_ylabel(r'$\Delta E$ [mRy]')
    ax.set_xlabel('Iteración')

def plot_force_rlx(forces, ax=None, **kwargs):
    """
    Plot difference of the force calculated at each iteration
    and the minimum force obtained at the end of the optimization.
    Args:
    - forces: array of energies
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure

    x = np.arange(len(forces))
    y = forces - forces.min()
    ax.plot(x, y*1e3, **kwargs)
    ax.set_ylabel(r'Fuerza neta [mRy/$a_0$]')
    ax.set_xlabel('Iteración')

def plot_dos_total(output, elements=True, ax=None, **kwargs):
    """
    Plot total DOS and optionally the projected DOS
    per chemical species.
    Args:
    - output: directory with extracted arrays of dos and pdos
    - elements: True or False, plot projected DOS
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure

    energy = output['energy']
    dos = output['dos']

    ax.plot(energy, dos, color='k', label='Total', **kwargs)
    if elements:
        for key in output:
            if key.count('_')==1:
                pdos = output[key]
                ax.plot(energy, pdos, label=key[5:], **kwargs)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Density of states')
    ax.legend()

def plot_dos_species(output, specie, orbitals=True, ax=None, **kwargs):
    """
    Plot projected DOS per chemical species and optionally 
    the projected DOS per orbital.
    Args:
    - output: directory with extracted arrays of dos and pdos
    - specie: chemical element to plot
    - orbitals: True or False, plot projected DOS
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots()
    else:          fig = ax.figure

    energy = output['energy']
    dos = output[f'pdos_{specie}']
    ax.plot(energy, dos, color='k', label=specie, **kwargs)
    if orbitals:
        for key in output:
            if key.count('_')==2 and key.startswith(f'pdos_{specie}'):
                pdos = output[key]
                ax.plot(energy, pdos, label=key[-1], **kwargs)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Projected density of states')
    ax.legend()

def plot_bands(output, ax=None, **kwargs):
    """
    Plot energy bands.
    Args:
    - output: directory with extracted arrays of energy bands
    - ax=None: optional
    - kwargs: for line plots
    """

    if ax is None: fig, ax = plt.subplots(figsize=(5,10))
    else:          fig = ax.figure

    k = output['kpoints']
    vBand = output['valence_band']
    cBand = output['conduction_band']

    # valence band
    for band in range(len(vBand)):
        ax.plot(k, vBand[band], color='k', **kwargs)
    # top of valence band
    idxs = np.where(vBand==vBand.max())
    i, j = idxs[0], idxs[1]
    ax.scatter(k[j], vBand[idxs], marker='o')

    # conduction band
    for band in range(len(cBand)):
        ax.plot(k, cBand[band], color='k', **kwargs)
    # bottom of valence band
    idxs = np.where(cBand==cBand.min())
    i, j = idxs[0], idxs[1]
    ax.scatter(k[j], cBand[idxs], marker='o')

    # high-symmetry points
    location = output['high_symmetry_points']
    style = dict(linewidth=0.75, color='gray', linestyle=':', zorder=1)
    for loc in location[1:-1]:
        ax.axvline(loc, **style)

    ax.axhline(0, **style)
    ax.axhline(cBand[i[0], j[0]], **style)
    ax.set_xticks([])
    ax.set_xlim([np.min(location), np.max(location)])
    ax.set_ylabel('Energía - Energía de Fermi [eV]')
    ax.minorticks_off()