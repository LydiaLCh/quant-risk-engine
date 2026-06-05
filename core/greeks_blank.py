import math 
from scipy.stats import norm
from black_scholes import d1, d2, bs_call

# Delta: the derivative df/dS of the BSM call formula / put formula 
def delta(S, K, T, r, sigma, option_type = "call"): 
    if option_type == "call":
        return norm.cdf(d1(S,K,T,r, sigma))
    else: 
        return norm.cdf(d1(S,K,T,r,sigma))-1 

print(delta(42,40,0.5,0.1,0.2))

#Gamma: the rate of change of delta per unit stock / share, equal to the standard normal PDF
def gamma(S,K,T,r, sigma): 
    return norm.pdf(d1(S,K,T,r,sigma)) / (S * sigma *math.sqrt(T))

print( f"{gamma(42,40,0.5,0.1,0.2):.4f}" )

#Vega: the change in the option price per % point of volatility 
def vega(S,K,T,r,sigma): 
    return (S * math.sqrt(T) * norm.pdf(d1(S,K,T,r,sigma)))/ 100

print( f"{vega(42,40,0.5,0.1,0.2):.8f}")

# Theta: the rate of change of the option price with respect to time passing 
def theta(S,K,T,r,sigma, option_type = "call"): 
    term_1 = -S * (norm.pdf(d1(S,K,T,r,sigma)) * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        term_2 = -r * K* math.exp(-r*T)* norm.cdf(d2(S,K,T,r,sigma))
    else: 
        term_2 = r * K* math.exp(-r*T)* norm.cdf(-d2(S,K,T,r,sigma))
    return (term_1 + term_2) / 365 

print(f"{theta(42,40,0.5,0.1,0.2)}:.4f")

# Rho: the rate of change of the option price with respect to the risk-free rate r
def rho(S,K,T,r,sigma, option_type = "call"): 
    if option_type == "call": 
        rho = (K*T*math.exp(-r*T) * norm.cdf(d2(S,K,T,r,sigma))) / 100
    else: 
        rho = -(K*T*math.exp(-r*T) * norm.cdf(-d2(S,K,T,r,sigma))) / 100
    return rho 

print(f"{rho(42,40,0.5,0.1,0.2):.4f}")

#Verification test for all Greeks using the BSM PDE relationship 

S,K,T,r,sigma = 42, 40, 0.5, 0.10, 0.20
f = bs_call(S,K,T,r,sigma)
lhs = theta(S,K,T,r,sigma)*365 + r*S* delta(S,K,T,r,sigma) + 0.5 * sigma**2 * S**2 * gamma(S,K,T,r,sigma)
rhs = r*f

assert abs(lhs - rhs) < 1e-4 