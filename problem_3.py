import numpy as np
import matplotlib.pyplot as plt


def steering_vec(
    wavelength,  # m
    r: np.ndarray,  # (num_element,) distance, m
    theta,  # degree
    num_element,
):
    a = np.zeros(num_element, dtype=np.complex_)
    for n in range(num_element):
        a[n] = np.exp(-2j * np.pi * r[n] * np.sin(theta * np.pi / 180) / wavelength)

    return a


def manifold_matrix(
    wavelength,  # m
    r: np.ndarray,  # (num_element,) distance, m
    theta: np.ndarray,  # (num_angle,) degree
    num_element,
):
    num_theta = len(theta)
    s = np.zeros((num_element, num_theta), dtype=np.complex_)
    for n in range(num_theta):
        s[:, n] = steering_vec(wavelength, r, theta[n], num_element)

    return s


def main():
    r = np.loadtxt("data/problem_3/array_elem_pos_mm.txt", dtype=np.double) / 1e3
    # s = np.loadtxt("data/problem_3/s_single_target.txt", dtype=np.complex_)
    s = np.loadtxt("data/problem_3/s_multi_target.txt", dtype=np.complex_)
    wavelength = 0.00389  # m
    theta_fov = 120  # degree
    n_grid = 480
    inc = theta_fov / n_grid

    theta = np.arange(n_grid + 1) * inc - 60
    num_element = len(s)
    # r = np.arange(num_element) * wavelength / 2
    map = manifold_matrix(wavelength, r, theta, num_element)
    mag = 20*np.log10(np.abs((map.conj().T) @ s))

    plt.figure()
    plt.semilogy(theta, mag)
    plt.grid(True)
    plt.xlabel("DoA(\u00b0)")
    plt.ylabel("Magnitude(dB)")
    plt.show()


if __name__ == "__main__":
    main()
