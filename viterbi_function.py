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

    # THE FOLLOWING SECTION MUST BE USED WHEN RUNNING THE EXAMPLE IN sim_func.py
    counter = 0
    # Count the couples of states 6&5 to check if the system is at high risk
    # counter = 0
    # most_likely_sequence = [x+1 for x in most_likely_sequence] # Since the indeces go from 0 to 5, it is better to increase them by 1 for clarity
    # for i in range(len(most_likely_sequence)-1):
    #     if most_likely_sequence[i] == 6 and most_likely_sequence[i+1] == 5:
    #         counter = counter + 1
        

    return most_likely_sequence, final_prob, counter

