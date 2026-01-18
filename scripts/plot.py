def get_plot_dispersion_curve():
    if material == 'diamond' or material == 'hbn':
        # Plotting logic
        fig = plt.figure(figsize=(9, 5))

        # convert to thz 
        unit_conversion = 241.798
        emax_ev = max(
            np.max(dos.get_energies()),
            np.max(np.asarray(bs.energies))
        ) * 1.1  # 3% headroom

        # Band Structure Axis
        ax = fig.add_axes([0.12, 0.1, 0.60, 0.8])
        bs.plot(ax=ax, emin=0.0, emax=emax_ev)

        # Convert Y-ticks to THz for readability
        ticks = ax.get_yticks()
        ax.set_yticklabels([f"{t * unit_conversion:.0f}" for t in ticks])
        ax.set_ylabel("Frequency (THz)", fontsize=14)

        # DOS Axis
        dosax = fig.add_axes([0.75, 0.1, 0.20, 0.8])
        dosax.fill_between(
            dos.get_weights(),
            dos.get_energies() * unit_conversion, # Scale energies to THz
            y2=0,
            color='grey', alpha=0.5
        )

        dosax.set_ylim(0, emax_ev*unit_conversion) # Now using the THz ceilings
        dosax.set_yticks([])
        dosax.set_xlabel('DOS', fontsize=14)
        fig.suptitle(f'{material} Phonon Dispersion and DOS', fontsize=16, y = 0.98)
        fig.savefig(f'{material}_Phonon.png', dpi=300)
    else:
        pass
    pass

get_plot_dispersion_curve()

EV_TO_THz = 241.79893 # 1 eV = 241.79893 THz (E = h nu)
THz_TO_CM1 = 33.35641 # 1 THz ≈ 33.35641 cm^-1

def compute_phonon_dos(ph, kpts=(12, 12, 12), npts=4000, width_thz=0.25):
    """
    Returns (freqs_thz, dos_states_per_thz).

    ph: an ASE Phonons object AFTER ph.run() and ph.read(...)
    kpts: Monkhorst-Pack grid used for DOS integration over the BZ
    width_thz: Gaussian broadening in THz (more intuitive than eV)
    """
    width_ev = width_thz / EV_TO_THz

    dos = ph.get_dos(kpts=kpts).sample_grid(npts=npts, width=width_ev)

    energies_ev = np.asarray(dos.get_energies())
    weights_per_ev = np.asarray(dos.get_weights())

    # Convert to frequency axis
    freqs_thz = energies_ev * EV_TO_THz

    # If plotting vs THz, convert DOS units: g(ν) = g(E) * dE/dν = g(E) / (dν/dE) = g(E) / EV_TO_THz
    weights_per_thz = weights_per_ev / EV_TO_THz

    # Guard against tiny negative frequencies (numerical noise)
    mask = freqs_thz >= -1e-6
    freqs_thz = freqs_thz[mask]
    weights_per_thz = weights_per_thz[mask]
    freqs_thz = np.clip(freqs_thz, 0.0, None)

    return freqs_thz, weights_per_thz

def plot_phonon_dos(freqs_thz, dos_states_per_thz, out_png,
                    title=None, emax_thz=None, dpi=450):
    """
    High-quality single-panel DOS plot: DOS on x-axis, frequency on y-axis (THz),
    with a secondary y-axis in cm^-1.
    """
    if emax_thz is None:
        emax_thz = float(np.max(freqs_thz)) * 1.02

    fig, ax = plt.subplots(figsize=(4.6, 6.2), constrained_layout=True)

    # Curve + filled area (no explicit color set; matplotlib defaults)
    ax.plot(dos_states_per_thz, freqs_thz, linewidth=1.2)
    ax.fill_betweenx(freqs_thz, 0.0, dos_states_per_thz, alpha=0.25)

    ax.set_xlabel("Phonon DOS (states / THz)")
    ax.set_ylabel("Frequency (THz)")
    ax.set_ylim(0.0, emax_thz)
    ax.set_xlim(left=0.0)

    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)

    if title:
        ax.set_title(title)

    # Secondary axis in cm^-1
    ax2 = ax.twinx()
    y0, y1 = ax.get_ylim()
    ax2.set_ylim(y0 * THz_TO_CM1, y1 * THz_TO_CM1)
    ax2.set_ylabel(r"Frequency (cm$^{-1}$)")
    ax2.tick_params(direction="in", top=True, right=True)

    fig.savefig(out_png, dpi=dpi)
    
freqs_thz, g_thz = compute_phonon_dos(ph, kpts=(12,12,12), npts=5000, width_thz=0.25)
plot_phonon_dos(freqs_thz, g_thz,
                out_png=f"{material}_DOS.png",
                title=f"{material} phonon DOS | k={12}^3 | σ=0.25 THz",
                emax_thz=50.0)
