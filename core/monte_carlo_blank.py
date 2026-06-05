import math 
import random 
from black_scholes import bs_call

def mc_call_price(S,K,T,r,sigma, n_paths):
    random.seed(42) #reproducibility, sets the starting point for the random number generator 
    payoffs = []
    for _ in range(n_paths):
        
        # Step 1: draw from the normal distribution N(0,1)
        Z = random.gauss(0,1)
        
        # Step 2: simulate S_T using the logonormal formula 
        S_T = S * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
        
        # Step 3: compute payoff 
        payoff = max(S_T - K, 0)

        payoffs.append(payoff)
    # Step 4: average all the payoffs over n_paths iterations and discount the time decay
    return   math.exp(-r *T) * (sum(payoffs) / n_paths)

# Verification test: check that monte carlo price is close to BSM analytical price: within 5 cents for 100k paths

# compute both prices
mc = mc_call_price(42, 40, 0.5, 0.10, 0.20, 100_000)
bs = bs_call(42, 40, 0.5, 0.10, 0.20)

# verify they agree within 5 cents 
assert abs(bs - mc)< 0.05, f"MC: {mc:.4f}, BS: {bs:.4f}"