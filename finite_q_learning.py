############################################################################################

import numpy as np
from tqdm import tqdm 
from scipy.optimize import minimize
import copy

# Robust Q learning (in finite spaces) #





 

def finite_q_learning(X, A, r, P_0, alpha, x_0, k_0, eps_greedy = lambda t: 0.05, Nr_iter = 1000, gamma_t_tilde = lambda t: 1/(t+1), Q_0 = None, save_Qt = False):
    """
    Parameters
    ----------
    X : numpy.ndarray
        A list or numpy array containing all states
    A : numpy.ndarray
        A list or numpy array containing all actions
    r : function
        Reward function r(x,a,y) depending on state-action-state.
    P_0 : numpy.ndarray
        P_0(x,a) list or numpy array of functions that creates a new random variabe in dependence of state and action
    alpha : float
        Discounting rate.
    x_0 : numpy.ndarray
        the initial state.
    eps_greedy : float, optional
        Parameter for the epsilon greedy policy. The default is 0.05.
    Nr_iter : int, optional
        Number of Iterations. The default is 1000.
    gamma_t_tilde : function, optional
        learning rate. The default is lambda t: 1/(t+1).
    Q_0 : matrix, optional
        Initial value for the Q-value matrix. The default is None.

    Returns
    -------
    matrix
        The Q value matrix.
    """

    rng = np.random.default_rng()

    # Initialize with Q_0 (if any)
    Q = np.zeros([len(X), len(A)])
    if Q_0 is not None:
        Q = Q_0
    
    if save_Qt:
        Qt = []
    
    # Initialize the Visits matrix
    Visits = np.zeros([len(X), len(A)])
    #Indic  = np.zeros([len(X), len(A)])
    
    # Initialize the empirical distribution matrix: emp_distribution[k, x, a, y] counts transitions
    emp_distribution = np.zeros([len(P_0), len(X), len(A), len(X)])

    # Catch A and X as lists type
    if np.ndim(A) > 1:
        A_list = A
    else:
        A_list = np.array([[a] for a in A])
    if np.ndim(X) > 1:
        X_list = X
    else:
        X_list = np.array([[x] for x in X])

    # Functions that catch the index of a in A
    def a_index(a):
        return np.flatnonzero((a == A_list).all(1))[0]
    def x_index(x):
        return np.flatnonzero((x == X_list).all(1))[0]
    
    # Define the f function
    def f(x, a, y):
        return r(x, a, y) + alpha * np.max(Q[x_index(y), :])
    
    # Vectorized f function for computing expected values
    def f_vectorized(x, a, Q_max_values):
        # Compute r(x, a, y) + alpha * max(Q[y, :]) for all y at once
        if np.ndim(X) > 1:
            rewards = np.array([float(r(x, a, X_list[y_ind])) for y_ind in range(len(X))])
        else:
            rewards = np.array([float(r(x, a, X[y_ind])) for y_ind in range(len(X))])
        return rewards + alpha * Q_max_values

    # Define the a_t function
    def a_t(y):
        eps_bound = eps_greedy(t)
        unif      = np.random.uniform(0)
        return (unif > eps_bound) * A[np.argmax(Q[x_index(y), :])] + (unif <= eps_bound) * rng.choice(A)
        
    
    
    # Set initial value
    Y_0 = [x_0 for P in P_0] # The markov decision processes
    # Iterations of Q and Visits and States
    for t in tqdm(range(Nr_iter)):
        
        A_0 = [a_t(y_0) for y_0 in Y_0]  # Get the actions according to the epsilon greedy policy

        x, a  = Y_0[k_0], A_0[k_0]

        Y_1 = [P_0[k](Y_0[k], A_0[k]) for k in range(len(Y_0))] # Update the Markov decision processes

        x_ind, a_ind = x_index(x), a_index(a)

        # Update the empirical distribution for all measures
        Y_1_indices = np.array([x_index(Y_1[k]) for k in range(len(P_0))])
        emp_distribution[np.arange(len(P_0)), x_ind, a_ind, Y_1_indices] += 1

        # Compute expected values w.r.t. empirical distribution for each measure k
        if Visits[x_ind, a_ind] > 0:
            # Pre-compute Q values for all states (for vectorized computation)
            Q_max_values = np.max(Q, axis=1)
            # Pre-compute f values for all states (same for all measures k)
            f_values = f_vectorized(x, a, Q_max_values)
            
            # Vectorized computation of expected values for all measures
            expected_values = np.zeros(len(P_0))
            for k in range(len(P_0)):
                emp_counts = emp_distribution[k, x_ind, a_ind, :]
                total_count = np.sum(emp_counts)
                if total_count > 0:
                    emp_probs = emp_counts / total_count
                    # Vectorized expected value computation
                    expected_values[k] = float(np.sum(emp_probs * f_values))
                else:
                    expected_values[k] = f(x, a, Y_1[k])
        else:
            # If no visits yet, use the immediate value as fallback
            expected_values = np.array([f(x, a, Y_1[k]) for k in range(len(P_0))], dtype=float)
        
        k_0 = np.argmin(expected_values) # Get the worst case index based on expected values
        y_1 = Y_1[k_0] # Get the next state
        
        Q_old_val = Q[x_ind, a_ind]

        # Do the update of Q
        Q[x_ind, a_ind] = Q_old_val +  gamma_t_tilde(Visits[x_ind, a_ind]) * (f(x, a, y_1) - Q_old_val)
        Visits[x_ind, a_ind] += 1 # Update the visits matrix
        if save_Qt:
            Qt.append(copy.deepcopy(Q))

        # Update the Markov decision processes
        Y_0 = [y_1 for P in P_0]
    
    if save_Qt:
        return Q, Visits, Qt
    else:
        return Q, Visits 


def finite_q_learning_pknown(X, A, r, P_0, p_0, alpha, x_0, k_0, eps_greedy = lambda t: 0.05, Nr_iter = 1000, gamma_t_tilde = lambda t: 1/(t+1), Q_0 = None, save_Qt = False):
    
    rng = np.random.default_rng()

    # Initialize with Q_0 (if any)
    Q = np.zeros([len(X), len(A)])
    if Q_0 is not None:
        Q = Q_0
    
    if save_Qt:
        Qt = []
    
    # Initialize the Visits matrix
    Visits = np.zeros([len(X), len(A)])

    # Catch A and X as lists type
    if np.ndim(A) > 1:
        A_list = A
    else:
        A_list = np.array([[a] for a in A])
    if np.ndim(X) > 1:
        X_list = X
    else:
        X_list = np.array([[x] for x in X])

    # Functions that catch the index of a in A
    def a_index(a):
        return np.flatnonzero((a == A_list).all(1))[0]
    def x_index(x):
        return np.flatnonzero((x == X_list).all(1))[0]
    
    # ============ PRECOMPUTE TRANSITION PROBABILITIES ============
    # Shape: [K, len(X), len(A), len(X)]
    # prob_matrix[k, x_ind, a_ind, y_ind] = p_0[k](X[x_ind], A[a_ind], X[y_ind])
    prob_matrix = np.zeros((len(P_0), len(X), len(A), len(X)))
    for k in range(len(P_0)):
        for x_idx, x in enumerate(X):
            for a_idx, a in enumerate(A):
                for y_idx, y in enumerate(X):
                    prob_matrix[k, x_idx, a_idx, y_idx] = float(p_0[k](x, a, y))
    # =============================================================
    
    # Define the f function
    def f(x, a, y):
        return r(x, a, y) + alpha * np.max(Q[x_index(y), :])
    
    # Vectorized f function for computing expected values
    def f_vectorized(x, a, Q_max_values):
        if np.ndim(X) > 1:
            rewards = np.array([float(r(x, a, X_list[y_ind])) for y_ind in range(len(X))])
        else:
            rewards = np.array([float(r(x, a, X[y_ind])) for y_ind in range(len(X))])
        return rewards + alpha * Q_max_values

    # Define the a_t function
    def a_t(y):
        eps_bound = eps_greedy(t)   
        unif      = np.random.uniform(0)
        return (unif > eps_bound) * A[np.argmax(Q[x_index(y), :])] + (unif <= eps_bound) * rng.choice(A)
    
    # Set initial value
    Y_0 = [x_0 for P in P_0]
    
    # Iterations of Q and Visits and States
    for t in tqdm(range(Nr_iter)):
        
        A_0 = [a_t(y_0) for y_0 in Y_0]
        x, a  = Y_0[k_0], A_0[k_0]
        Y_1 = [P_0[k](Y_0[k], A_0[k]) for k in range(len(Y_0))]

        x_ind, a_ind = x_index(x), a_index(a)

        # Compute expected values w.r.t. known probabilities
        if Visits[x_ind, a_ind] > 0:
            Q_max_values = np.max(Q, axis=1)
            f_values = f_vectorized(x, a, Q_max_values)
            
            # ============ OPTIMIZED EXPECTED VALUE COMPUTATION ============
            # Simply use precomputed probabilities with matrix multiplication
            expected_values = prob_matrix[:, x_ind, a_ind, :] @ f_values  # Shape: (len(P_0),)
            # ==============================================================
        else:
            expected_values = np.array([f(x, a, Y_1[k]) for k in range(len(P_0))], dtype=float)

        k_0 = np.argmin(expected_values)
        y_1 = Y_1[k_0]
        
        Q_old_val = Q[x_ind, a_ind]
        Q[x_ind, a_ind] = Q_old_val +  gamma_t_tilde(Visits[x_ind, a_ind]) * (f(x, a, y_1) - Q_old_val)
        Visits[x_ind, a_ind] += 1
        
        if save_Qt:
            Qt.append(copy.deepcopy(Q))

        Y_0 = [y_1 for P in P_0]
    
    if save_Qt:
        return Q, Visits, Qt
    else:
        return Q, Visits 
    
def finite_q_value_iteration(X, A, r, P_prob_list, alpha, Q_0=None, max_iter=1000, tolerance=1e-6):
    """
    Finite Q-value iteration with robust Bellman operator.
    
    Computes Q* using the robust Bellman operator:
    Q(x,a) = min_{k} E_{P_k(·|x,a)} [r(x,a,Y) + α max_{a'} Q(Y,a')]
    """
    
    # Build index functions (same as before)
    if np.ndim(A) > 1:
        A_list = A
    else:
        A_list = np.array([[a] for a in A])
    if np.ndim(X) > 1:
        X_list = X
    else:
        X_list = np.array([[x] for x in X])
    
    def a_index(a):
        return np.flatnonzero((a==A_list).all(1))[0]
    def x_index(x):
        return np.flatnonzero((x==X_list).all(1))[0]
    
    # Initialize Q-values
    if Q_0 is None:
        Q_values = np.zeros((len(X), len(A)))
    else:
        Q_values = copy.deepcopy(Q_0)
    
    K = len(P_prob_list)
    
    for iteration in tqdm(range(max_iter)):
        Q_old = copy.deepcopy(Q_values)
        
        # Precompute max Q-values for all states
        max_Q_old = np.max(Q_old, axis=1)  # Shape: (len(X),)
        
        # Update Q-values for each state-action pair
        for x_idx, x in enumerate(X):
            for a_idx, a in enumerate(A):
                # Compute expected value under each model P_k
                expected_values = np.zeros(K)
                
                for k_idx, P_k_prob in enumerate(P_prob_list):
                    # Vectorized computation over all next states
                    expected_val = 0.0
                    for y_idx, y in enumerate(X):
                        prob_y = float(P_k_prob(x, a, y))  # Ensure float
                        
                        if prob_y > 1e-10:  # Numerical stability threshold
                            reward = float(r(x, a, y))
                            bellman_val = reward + alpha * max_Q_old[y_idx]
                            expected_val += prob_y * bellman_val
                    
                    expected_values[k_idx] = expected_val
                
                # Robust Bellman operator: minimum over models
                Q_values[x_idx, a_idx] = np.min(expected_values)
        
        # Check convergence
        max_diff = np.max(np.abs(Q_values - Q_old))
        if max_diff < tolerance:
            print(f"✓ Converged after {iteration + 1} iterations (max diff: {max_diff:.2e})")
            return Q_values, iteration + 1
    
    print(f"⚠ Max iterations ({max_iter}) reached. Max diff: {max_diff:.2e}")
    return Q_values, max_iter