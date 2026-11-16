"""Analysis of LED I/V and light output data.
 
Reads several repeat runs, averages them (uncertainty = half the range
between runs), fits straight lines to chosen regions and makes four plots:
I vs V, ln(I) vs V, ln(I) vs ln(sqrt(L)) and efficiency vs I.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Optional, Sequence
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
 
 
@dataclass
class Measurement:
    """Mean and uncertainty at each point, averaged over runs."""
    mean: np.ndarray
    unc: np.ndarray
 
 
@dataclass
class LEDData:
    """Averaged voltage (V), current (A) and photodiode output (V)."""
    voltage: Measurement
    current: Measurement
    sensor: Measurement
 
 
@dataclass
class FitResult:
    """Straight line fit y = m*x + c. x and y_fit only include the points used."""
    gradient: float
    intercept: float
    gradient_err: float
    intercept_err: float
    x: np.ndarray
    y_fit: np.ndarray
 
    def label(self, name: str = "Linear fit") -> str:
        return (f"{name} (m={self.gradient:.4f}±{self.gradient_err:.4f}, "
                f"c={self.intercept:.4f})")
 
 
@dataclass
class FitRegion:
    """Which points to fit, plus the legend name and line colour."""
    index_slice: slice
    name: str = "Linear fit"
    colour: str = "red"
 
 
@dataclass
class AnalysisConfig:
    """Settings for one analysis run.
 
    Files are read as file_prefix0, file_prefix1, ... up to n_runs.
    light_start is the first point used for the light plots (to skip the
    noisy low current points). light_fit_start is the first point of the
    ln(I) vs ln(sqrt(L)) fit, counted in the full data set.
    """
    file_prefix: str = "i_v_data_"
    n_runs: int = 4
    led_name: str = "red LED"
    separator: str = "\t"
    voltage_col: str = "Diode Voltage"
    current_col: str = "Diode Current"
    sensor_col: str = "Sensor Output (V)"
    iv_fit: FitRegion = field(
        default_factory=lambda: FitRegion(slice(-3, None), "Linear fit", "red"))
    ln_iv_fits: Sequence[FitRegion] = field(default_factory=lambda: [
        FitRegion(slice(1, 10), "Linear fit 1", "red"),
        FitRegion(slice(14, 30), "Linear fit 2", "orange"),
    ])
    light_start: int = 33
    light_fit_start: int = 37
 
 
# Loading
 
def load_run(filename: str, config: AnalysisConfig):
    """Read one file and return (voltage, current, sensor) arrays."""
    df = pd.read_csv(filename, sep=config.separator)
    df.columns = df.columns.str.strip()
    return tuple(
        df[col].to_numpy(dtype=float)
        for col in (config.voltage_col, config.current_col, config.sensor_col)
    )
 
 
def average_runs(runs: Sequence[np.ndarray]) -> Measurement:
    """Mean over runs, with half the max-min range as the uncertainty."""
    stacked = np.vstack(runs)
    return Measurement(np.mean(stacked, axis=0), np.ptp(stacked, axis=0) / 2)
 
 
def load_led_data(config: AnalysisConfig) -> LEDData:
    """Load every run and average them."""
    volts, currents, sensors = [], [], []
    for n in range(config.n_runs):
        v, i, s = load_run(f"{config.file_prefix}{n}", config)
        volts.append(v)
        currents.append(i)
        sensors.append(s)
    return LEDData(average_runs(volts), average_runs(currents), average_runs(sensors))
 
 
# Fitting
 
def linear_model(x, m, c):
    return m * x + c
 
 
def ln_with_uncertainty(values, unc, power=1.0):
    """Return power*ln(values) and its uncertainty, power*unc/values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return power * np.log(values), abs(power) * unc / values
 
 
def fit_line(x, y, sigma=None, weighted_params=False) -> Optional[FitResult]:
    """Fit a straight line, ignoring nan/inf points.
 
    If sigma is given, points with sigma <= 0 are dropped too and the errors
    on m and c come from a weighted curve_fit. m and c themselves come from
    an unweighted polyfit unless weighted_params is True. Returns None if
    fewer than 2 points are left.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        valid &= np.isfinite(sigma) & (sigma > 0)
    if valid.sum() < 2:
        return None
 
    x, y = x[valid], y[valid]
    s = sigma[valid] if sigma is not None else None
 
    m, c = np.polyfit(x, y, deg=1)
    m_err = c_err = np.nan
    try:
        popt, pcov = curve_fit(linear_model, x, y, sigma=s,
                               absolute_sigma=s is not None)
        m_err, c_err = np.sqrt(np.diag(pcov))
        if weighted_params and s is not None:
            m, c = popt
    except (RuntimeError, ValueError):
        pass  # errors stay as nan
 
    return FitResult(m, c, m_err, c_err, x, linear_model(x, m, c))
 
 
def fit_region(x, y, sigma, region: FitRegion, **kwargs) -> Optional[FitResult]:
    """Run fit_line on the points picked out by a FitRegion."""
    sl = region.index_slice
    return fit_line(x[sl], y[sl], None if sigma is None else sigma[sl], **kwargs)
 
 
# Plots
 
def plot_iv(data: LEDData, config: AnalysisConfig) -> tuple[Figure, Optional[FitResult]]:
    """I vs V with a linear fit over config.iv_fit."""
    fig, ax = plt.subplots()
    ax.errorbar(data.voltage.mean, data.current.mean,
                yerr=data.current.unc, xerr=data.voltage.unc, fmt="o")
 
    fit = fit_region(data.voltage.mean, data.current.mean, data.current.unc,
                     config.iv_fit)
    if fit is not None:
        ax.plot(fit.x, fit.y_fit, color=config.iv_fit.colour, linewidth=2,
                alpha=0.7, label=fit.label(config.iv_fit.name))
        ax.legend()
 
    ax.set_xlabel("Voltage / V")
    ax.set_ylabel("Current / A")
    ax.set_title(f"I/V characteristics for a {config.led_name}")
    return fig, fit
 
 
def plot_ln_iv(data: LEDData, config: AnalysisConfig) -> tuple[Figure, list]:
    """ln(I) vs V with one linear fit per region in config.ln_iv_fits."""
    ln_i, ln_i_unc = ln_with_uncertainty(data.current.mean, data.current.unc)
 
    fig, ax = plt.subplots()
    ax.errorbar(data.voltage.mean, ln_i, yerr=ln_i_unc, xerr=data.voltage.unc,
                fmt="o", label="Data", markersize=5, capsize=3, ecolor="black")
 
    fits = []
    for region in config.ln_iv_fits:
        fit = fit_region(data.voltage.mean, ln_i, ln_i_unc, region)
        fits.append(fit)
        if fit is not None:
            ax.plot(fit.x, fit.y_fit, color=region.colour, linewidth=2,
                    alpha=0.7, label=fit.label(region.name))
 
    ax.set_xlabel("Voltage / V")
    ax.set_ylabel("ln(Current / A)")
    ax.set_title(f"ln(I) vs V characteristics for a {config.led_name}")
    ax.legend()
    return fig, fits
 
 
def plot_light_vs_current(data: LEDData, config: AnalysisConfig
                          ) -> tuple[Figure, Optional[FitResult]]:
    """ln(I) vs ln(sqrt(L)), taking L as the photodiode output."""
    start = config.light_start
    ln_sqrt_l, ln_sqrt_l_unc = ln_with_uncertainty(
        data.sensor.mean[start:], data.sensor.unc[start:], power=0.5)
    ln_i, ln_i_unc = ln_with_uncertainty(
        data.current.mean[start:], data.current.unc[start:])
 
    fit_from = config.light_fit_start - start
    fit = fit_line(ln_sqrt_l[fit_from:], ln_i[fit_from:], ln_i_unc[fit_from:])
 
    fig, ax = plt.subplots()
    ax.errorbar(ln_sqrt_l, ln_i, yerr=ln_i_unc, xerr=ln_sqrt_l_unc, fmt="o",
                label="Sensor data", markersize=5, capsize=3, ecolor="black")
    if fit is not None:
        ax.plot(fit.x, fit.y_fit, color="orange", linewidth=2, alpha=0.7,
                label=fit.label())
    ax.set_xlabel(r"ln(L$^{1/2}$)")
    ax.set_ylabel("ln(Current / A)")
    ax.set_title("ln(I) vs ln(L$^{1/2}$)")
    ax.legend()
    return fig, fit
 
 
def plot_efficiency(data: LEDData, config: AnalysisConfig) -> Figure:
    """Photodiode output / current against current (uncertainties added in quadrature)."""
    s = slice(config.light_start, None)
    sensor, sensor_unc = data.sensor.mean[s], data.sensor.unc[s]
    current, current_unc = data.current.mean[s], data.current.unc[s]
 
    efficiency = sensor / current
    efficiency_unc = efficiency * np.sqrt((sensor_unc / sensor) ** 2
                                          + (current_unc / current) ** 2)
 
    fig, ax = plt.subplots()
    ax.errorbar(current, efficiency, xerr=current_unc, yerr=efficiency_unc,
                ecolor="black", fmt="o")
    ax.set_xlabel("Current / A")
    ax.set_ylabel("Efficiency (L/I)")
    ax.set_title(f"Efficiency vs current for a {config.led_name}")
    return fig
 
 
def run_analysis(config: AnalysisConfig, show: bool = True) -> dict:
    """Load data, make all the plots, return the fits and figures in a dict."""
    data = load_led_data(config)
 
    fig1, iv_fit = plot_iv(data, config)
    fig2, ln_iv_fits = plot_ln_iv(data, config)
    fig3, light_fit = plot_light_vs_current(data, config)
    fig4 = plot_efficiency(data, config)
 
    if show:
        plt.show()
 
    return {"iv": iv_fit, "ln_iv": ln_iv_fits, "light": light_fit,
            "figures": [fig1, fig2, fig3, fig4]}
 
 
if __name__ == "__main__":
    run_analysis(AnalysisConfig(file_prefix="i_v_data_", n_runs=4))