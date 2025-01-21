import numpy as np
import matplotlib.pyplot as plt
from pyparsing import nums

def simulate(k, D, delta, warmup):
    """
    Simulates a sequence of states and observations for given HMM parameters.

    Args:
    - k: Length of the sequence (default 200)
    - D: Initial state probabilities (default [1, 0, 0, 0, 0, 0])
    - delta: Policy modifier (default 0)
    - warmup: If True, simulate 50 warm-up steps and return the result from step 51 (default False)

    Returns:
    - s_onehot: One-hot encoded state sequence
    - o_onehot: One-hot encoded observation sequence
    """

    B = np.array([
        [0.1, 0.9, 0, 0, 0, 0],
        [0.2, 0, 0.1 + delta, 0, 0, 0.7 - delta],
        [0, 0.5, 0, 0.5, 0, 0],
        [0, 0, 0.5, 0.5, 0, 0],
        [0, 0, 0.1, 0, 0.4, 0.5],
        [0, 0.2, 0, 0, 0.7, 0.1]
    ])

    A = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.4, 0.1, 0.1],
        [0.4, 0.4, 0.1, 0.1],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.1, 0.2, 0.6],
        [0.15, 0.1, 0.35, 0.4]
    ])

    total_steps = k + 50 if warmup else k

    s = np.zeros(total_steps, dtype=int)
    o = np.zeros(total_steps, dtype=int)

    # Initial state and observation
    s[0] = np.random.choice(6, p=D)
    o[0] = np.random.choice(4, p=A[s[0], :])

    # Simulate the sequence
    for t in range(1, total_steps):
        s[t] = np.random.choice(6, p=B[s[t - 1], :])
        o[t] = np.random.choice(4, p=A[s[t], :])

    # Apply warm-up logic
    if warmup:
        s = s[50:]
        o = o[50:]

    s_onehot = np.eye(6)[s]
    o_onehot = np.eye(4)[o]

    return s_onehot, o_onehot


def viterbi_algorithm(B, A, D, delta, observations):
    """
    Args:
        B (numpy.array): Matrix of state transition probabilities
                           B[i,j] = probability of transition from state i to state j
        A (numpy.array): Matrix that relates states and observations
                           A[i,j] = probability of observing j for state i
        D (numpy.ndarray): Initial probability vector of size (N_states,1).
                           D[i] = P(state_i at t=0).
        observations (numpy.ndarray): One-hot encoded observation matrix of size
                                       (N_observations x T), where each column represents
                                       an observation at time t.
    
    Returns:
            - most_likely_sequence (list): The most probable sequence of states.
            - final_prob (float): Probability of the most probable sequence.
            - counter (int): Number of observed state sequences of 6 & 5.
    """


    N_states = B.shape[0]
    T = observations.shape[1]  # Number of time steps

    # Initialize the DP table and backpointer table
    dp = np.zeros((N_states, T))  # Maximum probabilities
    psi = np.zeros((N_states, T), dtype=int)  # State transitions with max probability

    # Observation indices are derived from the one-hot encoding
    # takes the index of the observed state in each instant (each column representing a different time instant)
    obs_indices = np.argmax(observations, axis=0)

    # Initialization step (t=0)
    dp[:, 0] = D # initial probabilities
    psi[:, 0] = -1

    # Recursion step (t=1 to T-1)
    for t in range(1, T):
        for s in range(N_states):
            probabilities = dp[:, t-1] * B[:, s] * A[s, obs_indices[t]]
            dp[s, t] = np.max(probabilities) # take the max value of the probabilities computed
            psi[s, t] = np.argmax(probabilities) # save the coordinate of the state with max probability

    # Termination step: Identify the final state with the highest probability
    final_state = np.argmax(dp[:, -1])
    final_prob = dp[final_state, -1]

    # Retrieve the most likely sequence of states
    most_likely_sequence = [int(final_state)]
    for t in range(T-1, 0, -1):
        most_likely_sequence.insert(0, int(psi[most_likely_sequence[0], t]))

    # Count the couples of states 6&5 to check if the system is at high risk
    counter = 0
    most_likely_sequence = [x+1 for x in most_likely_sequence] # Since the indeces go from 0 to 5, it is better to increase them by 1 for clarity
    for i in range(len(most_likely_sequence)-1):
        if most_likely_sequence[i] == 6 and most_likely_sequence[i+1] == 5:
            counter = counter + 1
        

    return most_likely_sequence, final_prob, counter

#--------------------------------------------------------------------------------------------
k = 200
D = np.array([1, 0, 0, 0, 0, 0])
delta = 0.4
warmup = False
n = 100
m = 5
f = 0.8

B = np.array([
    [0.1, 0.9, 0, 0, 0, 0],
    [0.2, 0, 0.1 + delta, 0, 0, 0.7 - delta],
    [0, 0.5, 0, 0.5, 0, 0],
    [0, 0, 0.5, 0.5, 0, 0],
    [0, 0, 0.1, 0, 0.4, 0.5],
    [0, 0.2, 0, 0, 0.7, 0.1]
])

A = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.4, 0.4, 0.1, 0.1],
    [0.4, 0.4, 0.1, 0.1],
    [0.3, 0.3, 0.3, 0.1],
    [0.1, 0.1, 0.2, 0.6],
    [0.15, 0.1, 0.35, 0.4]
])
#warmup = True
# states, observations = simulate(k, D, delta,warmup)

# # Plotting the results
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ax1.imshow(states.T, aspect='auto', cmap='viridis')
# ax1.set_title('States')
# ax1.set_xlabel('Time step')
# ax1.set_ylabel('State')
# ax1.set_yticks(np.arange(6))
# ax1.set_yticklabels(['1', '2', '3', '4', '5', '6'])

# ax2.imshow(observations.T, aspect='auto', cmap='plasma')
# ax2.set_title('Observations')
# ax2.set_xlabel('Time step')
# ax2.set_ylabel('Observation')
# ax2.set_yticks(np.arange(4))
# ax2.set_yticklabels(['α', 'β', 'γ', 'μ'])

# plt.tight_layout()
# plt.show()

#-----------------------------------------------------------------------------
# DECODING
#-----------------------------------------------------------------------------

# Run the Viterbi algorithm
# observations = np.transpose(observations)
# sequence, prob, counter = viterbi_algorithm(D, delta, observations)

# print("Most likely sequence:", sequence)
# print("Probability of sequence:", prob)

bad_labels = 0
for i in range(n):
    _, observations = simulate(k, D, delta,warmup)
    observations = np.transpose(observations)
    sequence, prob, counter = viterbi_algorithm(B, A, D, delta, observations)
    print("Most likely sequence:", sequence)
    print("Probability of sequence:", prob)
    print("Counter: ", counter)
    if counter > m:
        # print("The system is at high risk\n")
        bad_labels = bad_labels + 1
    # else:
        # print("The system is at low risk\n")

# Check if operating conditions are risky
if bad_labels >= f*n:
    print(bad_labels, " bad labels found, risky operating conditions!\n")
else:
    print(bad_labels, " bad labels found, normal operating conditions.\n")
