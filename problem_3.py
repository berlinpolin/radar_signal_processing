import numpy as np


def steering_vec(
        wavelength, # m
        r:np.ndarray, # distance, m
        theta, # degree
        num_element
        ):
    a = np.zeros(num_element, dtype=np.complex_)
    for n in range(num_element):
        a[n] = np.exp(-2j*np.pi*r[n]*np.sin(theta)/wavelength)

    return a

def main():
    s = np.loadtxt('data/problem_3/s_single_target.txt', dtype=np.complex_)
    wavelength = 0.00389 # m
    theta_fov = 120 # degree

if __name__ == "__main__":
    main()