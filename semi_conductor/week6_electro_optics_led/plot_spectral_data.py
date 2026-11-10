import os
import re
import matplotlib.pyplot as plt
import numpy as np


def get_data(file_path):
    pattern = r"(\d+)ma_spectra_(\d+)\.txt"

    # Set up your figure canvas once before the loop starts
    plt.figure(figsize=(10, 6))

    for file in os.listdir(file_path):
        match = re.search(pattern, file)

        # Only process the file if it matches your regular expression
        if match:
            # Convert them to integers so you can use them mathematically
            current = int(match.group(1))
            time_step = int(match.group(2))

            # Safely join paths regardless of your operating system
            full_path = os.path.join(file_path, file)

            try:
                wavelength, counts = np.loadtxt(full_path, unpack=True)

                # Divide counts by time step as requested
                counts = counts / time_step

                # Slice data between 580 and 720nm
                mask = (wavelength >= 580) & (wavelength <= 720)

                # Plot data on the shared canvas
                plt.plot(
                    wavelength[mask],
                    counts[mask],
                    label=f"{current}mA",
                    linewidth=1.5,
                )

            except Exception as e:
                print(f"Skipping {file} due to an error reading data: {e}")

    # Set formatting elements after the loop finishes parsing files
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Counts / ms")
    plt.title("Intensity Spectroscopy for red LED (580nm-720nm)")

    # Force graph viewport to match your mask range exactly
    plt.xlim(580, 720)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    get_data("spectra_data")
