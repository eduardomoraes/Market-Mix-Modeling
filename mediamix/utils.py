import numpy as np
import pymc3 as pm # For pm.math if needed, though these are primarily numpy

def geometric_adstock(spend_array, decay_rate, L=None, normalize=False):
    """
    Calculates the adstock effect with a geometric decay.

    Parameters:
    ----------
    spend_array : np.ndarray or pm.TensorVariable
        A 1D array or PyMC tensor representing the spend for a channel over time.
    decay_rate : float or pm.TensorVariable
        The decay rate (between 0 and 1) for the adstock effect.
        A value of 0 means no carryover, 1 means full carryover (less common).
    L : int, optional
        The maximum length of the carryover effect (lookback window).
        If None, the carryover effect can extend to the beginning of the spend_array.
    normalize : bool, optional
        If True, normalizes the adstocked spend to maintain the same total sum as the original spend.
        This helps in interpreting the beta coefficients more directly.

    Returns:
    -------
    np.ndarray or pm.TensorVariable
        A 1D array or PyMC tensor of the same shape as spend_array, representing the adstocked spend.
    """
    # If inputs are PyMC tensors, use pm.math, else use np
    # This function as-is will be problematic if decay_rate is a PyMC variable
    # due to the loop. For PyMC, this would typically be implemented with pm.scan.
    # For now, this is a NumPy-based utility.
    
    adstocked_spend = np.zeros_like(spend_array, dtype=float)
    if not L: # if L is not specified, consider full length of spend_array for carryover
        L = len(spend_array)
    
    for t in range(len(spend_array)):
        for l_idx in range(min(t + 1, L)): # l_idx is the lag index
            adstocked_spend[t] += spend_array[t-l_idx] * (decay_rate ** l_idx)
            
    if normalize:
        sum_adstocked_spend = np.sum(adstocked_spend)
        if sum_adstocked_spend > 0:
            return adstocked_spend / sum_adstocked_spend * np.sum(spend_array)
        else: # handle case where sum is zero to avoid division by zero
            return adstocked_spend 
    return adstocked_spend

def logistic_saturation(spend_array, L, k, x0):
    """
    Applies a logistic saturation function to spend data.

    Parameters:
    ----------
    spend_array : np.ndarray or pm.TensorVariable
        A 1D array or PyMC tensor representing the (possibly adstocked) spend.
    L : float or pm.TensorVariable
        The maximum effect (saturation point or asymptote).
    k : float or pm.TensorVariable
        The steepness of the curve (growth rate).
    x0 : float or pm.TensorVariable
        The mid-point of the function (spend level at which the effect is L/2).

    Returns:
    -------
    np.ndarray or pm.TensorVariable
        A 1D array or PyMC tensor of the same shape as spend_array, 
        representing the saturated effect.
    """
    # This function can be translated to pm.math fairly directly:
    # return L / (1 + pm.math.exp(-k * (spend_array - x0)))
    return L / (1 + np.exp(-k * (spend_array - x0)))

if __name__ == '__main__':
    # Example Usage (NumPy)
    spend = np.array([10, 12, 15, 13, 16, 18, 20, 17, 14, 11], dtype=float)
    
    # Adstock
    decay = 0.6
    adstocked_spend_np = geometric_adstock(spend, decay, L=5, normalize=True)
    print("NumPy Adstocked Spend (L=5, normalized):\n", adstocked_spend_np)

    adstocked_spend_full_np = geometric_adstock(spend, decay, normalize=False)
    print("\nNumPy Adstocked Spend (full carryover, not normalized):\n", adstocked_spend_full_np)

    # Saturation
    # Assume adstocked_spend_full_np is the input for saturation
    L_sat = 100.0
    k_sat = 0.1
    x0_sat = np.median(adstocked_spend_full_np) # Example: mid-point at median of adstocked spend
    
    saturated_effect_np = logistic_saturation(adstocked_spend_full_np, L_sat, k_sat, x0_sat)
    print("\nNumPy Saturated Effect:\n", saturated_effect_np)

    # Example with small values to test normalization edge case
    spend_small = np.array([0.0, 0.0, 0.0])
    adstocked_small = geometric_adstock(spend_small, decay, normalize=True)
    print("\nNumPy Adstocked Spend (small values, normalized):\n", adstocked_small)
