import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def main():
    # load the dataset
    dataset = np.load('data/problem_2/radar_imaging_raw_data.npz')
    radar_raw_data_cube = np.squeeze(dataset['raw_tdm_frame']) # raw data cube
    f_s = dataset['adc_sampling_freq']                         # ADC sampling rate (Hz)
    num_samp_per_chirp = dataset['tdm_num_samples_per_chirp']  # number of samples per chirp
    num_chirp_per_frame = dataset['tdm_num_blocks']            # number of chirps per frame
    num_rx_antenna = dataset['n_rx']                           # number of RX antennas
    num_tx_antenna = dataset['n_tx']                           # number of TX antennas
    chirp_duration = dataset['tdm_chirp_duration']             # chirp duration (s)
    slew_rate = 9.994e6/1e-6                                   # chirp slew rate (Hz/s)
    c_air = 299702547.236                                      # speed of light in air (m/s)
    wavelength = c_air/dataset['carrier_freq']                 # carrier wavelength (m)
    antenna_spacing = wavelength/2                             # distance of adjacent antennas

    radar_raw_data_cube_range_velocity_fft = \
        np.fft.fft2(radar_raw_data_cube)
    radar_raw_data_cube_range_velocity_fft = \
        np.fft.fftshift(radar_raw_data_cube_range_velocity_fft, axes=(1,))
    radar_raw_data_cube_range_velocity_fft_mag_avg_db = \
        10*np.log10(np.mean(np.abs(radar_raw_data_cube_range_velocity_fft)**2, axis=0))
    
    freq_x = (np.fft.fftshift(np.fft.fftfreq(int(num_samp_per_chirp))) + 0.5) *\
                 num_samp_per_chirp # Frequency values for y-axis
    freq_y = np.fft.fftshift(np.fft.fftfreq(int(num_chirp_per_frame))) *\
                    num_chirp_per_frame  # Frequency values for x-axis
    freq_x = freq_x*f_s*c_air/(num_samp_per_chirp*2*slew_rate)
    freq_y = freq_y*wavelength/(num_chirp_per_frame*2*chirp_duration)

    print('Max. Range :', f_s*c_air/(2*slew_rate))
    print('Max. Velocity :', wavelength/(4*chirp_duration))

    plt.figure()
    plt.imshow(radar_raw_data_cube_range_velocity_fft_mag_avg_db, \
               extent=[freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]])
    plt.colorbar(label="Magnitude (log scale)")
    plt.xlabel("Range (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Range-Velocity Map")
    plt.show()


if __name__ == "__main__":
    main()