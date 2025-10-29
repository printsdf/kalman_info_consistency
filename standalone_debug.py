import numpy as np

# Helper functions
def k_gain(P, H, R):
    S = H @ P @ H.T + R
    return P @ H.T @ np.linalg.inv(S)

def joseph(P, K, H, R):
    I = np.eye(P.shape[0], dtype=P.dtype)
    return (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

def current_info_update(Y_prior, H, R):
    # This is what we currently have in info.py
    Y = Y_prior.astype(np.float64)
    H = H.astype(np.float64)
    R = R.astype(np.float64)
    
    # Ensure H is a row observation
    if H.ndim == 1:
        H = H.reshape(1, -1)
    elif H.ndim == 2 and H.shape[0] != 1 and H.shape[1] == 1:
        H = H.T
    
    R_inv = np.linalg.inv(R)
    return Y + H.T @ R_inv @ H

def correct_info_update_from_covariance(P, H, R):
    # First compute the correct Joseph update in covariance form
    K = k_gain(P, H, R)
    P_j = joseph(P, K, H, R)
    
    # Then convert to information form
    Y_j = np.linalg.inv(P_j)
    return Y_j

def amplified_case():
    P = np.array([[100.0, 99.0],
                  [99.0, 100.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    R = np.array([[0.001]], dtype=np.float64)
    return P, H, R

# Test the implementations
P, H, R = amplified_case()
K = k_gain(P, H, R)

# Joseph update in covariance form
P_j = joseph(P, K, H, R)
print("Joseph update (covariance):")
print(P_j)

# Current info update
Y_prior = np.linalg.inv(P)
Y_current = current_info_update(Y_prior, H, R)
P_from_current_info = np.linalg.inv(Y_current)
print("\nCurrent info update -> covariance:")
print(P_from_current_info)

# Correct info update (derived from Joseph)
Y_correct = correct_info_update_from_covariance(P, H, R)
print("\nCorrect info update (from Joseph):")
print(Y_correct)

# Calculate the difference
print(f"\nDifference in info matrices: {np.max(np.abs(Y_current - Y_correct)):.10f}")
print(f"Difference in covariance matrices: {np.max(np.abs(P_j - P_from_current_info)):.10f}")

# Let's also print the mathematical formulas to understand the issue
print("\nMathematical check:")
print(f"P shape: {P.shape}, H shape: {H.shape}, R shape: {R.shape}")
print(f"Y_prior shape: {Y_prior.shape}")

# Calculate the terms separately
R_inv = np.linalg.inv(R)
Ht_Rinv_H = H.T @ R_inv @ H
print(f"H.T @ R_inv @ H:\n{Ht_Rinv_H}")

# Let's try the correct formula for info update that should match Joseph
# The correct formula should account for the Kalman gain
# Y_post = Y_prior + H.T @ R_inv @ H - Y_prior @ H.T @ (H @ Y_prior @ H.T + R_inv)^(-1) @ H @ Y_prior
# Let's compute this
Y_prior_inv = P  # Since Y_prior = P^(-1)
term1 = Y_prior
term2 = H.T @ R_inv @ H
S = H @ Y_prior_inv @ H.T + R
S_inv = np.linalg.inv(S)
term3 = Y_prior @ H.T @ S_inv @ H @ Y_prior
Y_alternative = term1 + term2 - term3

print("\nAlternative info update formula:")
print(Y_alternative)
print(f"Difference from correct (Joseph): {np.max(np.abs(Y_alternative - Y_correct)):.10f}")

# Convert back to covariance
P_alternative = np.linalg.inv(Y_alternative)
print(f"\nAlternative info update -> covariance:")
print(P_alternative)
print(f"Difference from Joseph covariance: {np.max(np.abs(P_alternative - P_j)):.10f}")

# Let's also check if the Joseph update is PSD
w_j = np.linalg.eigvalsh(P_j)
print(f"\nJoseph eigenvalues: {w_j}")
print(f"Is Joseph PSD? {np.all(w_j > -1e-12)}")

# And the current info update -> covariance
w_current = np.linalg.eigvalsh(P_from_current_info)
print(f"\nCurrent info -> covariance eigenvalues: {w_current}")
print(f"Is current info -> covariance PSD? {np.all(w_current > -1e-12)}")

# And the alternative
w_alt = np.linalg.eigvalsh(P_alternative)
print(f"\nAlternative eigenvalues: {w_alt}")
print(f"Is alternative PSD? {np.all(w_alt > -1e-12)}")