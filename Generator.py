import random
import numpy as np
import matplotlib.pyplot as plt


# create_binary_samples:
# Given the data_size - the amount of samples we want, and the prob
# which is the probability for a neuron spike, this outputs a binary
# array which is supposed to be the "real" neuron data.
def create_binary_samples(data_size, prob=100):
    return [1 if random.randint(1, prob) == 1 else 0 for i in range(data_size)]


# get_noisy_bin_array:
# Takes the binary sample array and adds a gaussian noise to it -
# If the noise is over the threshold it could create a false positive
# or false negative if there was an actual spike in the original sample.
def get_noisy_bin_array(bin_arr, mu, sigma, threshold):
    noisy_bin_arr = np.zeros(len(bin_arr))
    for i in range(len(bin_arr)):
        noisy_bin_arr[i] = bin_arr[i] + random.choice(np.random.normal(mu, sigma, 1))
        # print(noisy_bin_arr[i])
        if noisy_bin_arr[i] > threshold:
            noisy_bin_arr[i] = 1
        else:
            noisy_bin_arr[i] = 0
    return [int(item) for item in noisy_bin_arr]


# get_delayed_bin_array:
# Takes the noisy sample array and adds a delay:
# Creates a gaussian random variable and adds it as a "delay".
# Can cause a backward or forward delay.
def get_delayed_bin_array(bin_arr, sigma=0.5):
    delayed_arr = [0 for i in range(len(bin_arr))]
    for i in range(len(bin_arr)):
        if bin_arr[i] == 0:
            continue
        delay = random.choice(np.random.normal(0, sigma, 1))
        if (i == 0 and delay < -0.5) or (i == len(bin_arr) - 1 or delay >= 0.5):
            continue
        delayed_arr[i + int(round(delay))] = 1
    return delayed_arr


# Given the binary neuron samples array, it creates an array of
# vascular activity samples that is affected by a gaussian noise.
# TODO: need to make the samples accumulative, and start of the activity should be before neuron sample.
def create_x_array(bin_arr, mu, sigma=0.2, dec=0.8):
    out_arr = [0.0 for i in range(len(bin_arr))]
    for i in range(len(bin_arr)):
        if bin_arr[i] == 1:
            rise_value = random.choice(np.random.normal(mu, sigma, 1))
            for j in range (i, len(out_arr)):
                out_arr[j] = out_arr[j] + rise_value * (dec ** (j - i))
            for j in range (i - 1, -1, -1):
                out_arr[j] = out_arr[j] + rise_value * (dec ** (2 * (i - j)))
    return out_arr


bin_array = create_binary_samples(100, 25)
print(bin_array)
noisy_bin_array = get_noisy_bin_array(bin_array, 0, 0.5, 0.5)
print(noisy_bin_array)
delayed_array = get_delayed_bin_array(noisy_bin_array)
print(delayed_array)
x_array = create_x_array(bin_array, 5)
print(x_array)
plt.plot(bin_array, c='r', ls='--', label='Reference')
plt.plot(x_array, c='b', label='noisy')
plt.show()
