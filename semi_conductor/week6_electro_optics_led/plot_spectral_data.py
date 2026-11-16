"""Plot LED emission spectra from a folder of spectrometer files.
 
Files are named like "20ma_spectra_100.txt": the first number is the LED
current in mA and the second is the integration time. Each file has two
columns, wavelength and counts.
"""
 
from __future__ import annotations
 
import os
import re
from dataclasses import dataclass
from typing import Optional
 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
 
DEFAULT_PATTERN = r"(\d+)ma_spectra_(\d+)\.txt"
 
 
@dataclass
class Spectrum:
    """One spectrum. counts is already divided by the integration time."""
    current: int
    time_step: int
    wavelength: np.ndarray
    counts: np.ndarray
 
 
def read_two_columns(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read wavelength and counts, ignoring any line that isn't two numbers.
 
    Unlike np.loadtxt this copes with headers, footers or odd extra rows in
    the file. Prints a message if any non-empty lines were dropped.
    """
    wavelength, counts, dropped = [], [], 0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                if len(parts) != 2:
                    raise ValueError
                w, c = float(parts[0]), float(parts[1])
            except ValueError:
                dropped += 1
                continue
            wavelength.append(w)
            counts.append(c)
 
    if dropped:
        print(f"{os.path.basename(path)}: ignored {dropped} lines that "
              "weren't two numbers")
    return np.array(wavelength), np.array(counts)
 
 
def load_spectrum(path: str, current: int, time_step: int,
                  wl_range: Optional[tuple[float, float]] = None) -> Spectrum:
    """Read one file, divide counts by time_step and cut to wl_range (nm)."""
    wavelength, counts = read_two_columns(path)
    counts = counts / time_step
 
    if wl_range is not None:
        mask = (wavelength >= wl_range[0]) & (wavelength <= wl_range[1])
        wavelength, counts = wavelength[mask], counts[mask]
 
    return Spectrum(current, time_step, wavelength, counts)
 
 
def load_spectra(folder: str, pattern: str = DEFAULT_PATTERN,
                 wl_range: Optional[tuple[float, float]] = None) -> list[Spectrum]:
    """Load every file in folder whose name matches pattern.
 
    The pattern needs two groups: current first, then time step. Files that
    don't match are ignored and files that can't be read are skipped with a
    message. Result is sorted by current.
    """
    spectra = []
    for name in os.listdir(folder):
        match = re.search(pattern, name)
        if not match:
            continue
        current, time_step = int(match.group(1)), int(match.group(2))
        try:
            spectra.append(load_spectrum(os.path.join(folder, name),
                                         current, time_step, wl_range))
        except Exception as e:
            print(f"Skipping {name}, couldn't read data: {e}")
 
    return sorted(spectra, key=lambda s: s.current)
 
 
def plot_spectra(spectra: list[Spectrum], led_name: str = "red LED",
                 wl_range: Optional[tuple[float, float]] = None,
                 count_unit: str = "ms") -> Figure:
    """Plot all spectra on one set of axes, one line per current."""
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for s in spectra:
        ax.plot(s.wavelength, s.counts, label=f"{s.current}mA", linewidth=1.5)
 
    title = f"Intensity spectroscopy for {led_name}"
    if wl_range is not None:
        title += f" ({wl_range[0]:g}nm-{wl_range[1]:g}nm)"
        ax.set_xlim(*wl_range)
 
    ax.set_xlabel("Wavelength / nm")
    ax.set_ylabel(f"Counts / {count_unit}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
 
 
def run_analysis(folder: str, pattern: str = DEFAULT_PATTERN,
                 wl_range: Optional[tuple[float, float]] = (580, 720),
                 led_name: str = "red LED", show: bool = True) -> Figure:
    """Load spectra from folder and plot them. Returns the figure."""
    spectra = load_spectra(folder, pattern, wl_range)
    fig = plot_spectra(spectra, led_name, wl_range)
    if show:
        plt.show()
    return fig
 
 
if __name__ == "__main__":
    run_analysis("spectra_data")
 