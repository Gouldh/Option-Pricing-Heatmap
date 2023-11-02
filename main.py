import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# -----------------------------------------------------------------------------------
# Author: Hunter Gould
# Date: 11/02/2023
# Description: This project involves the implementation of various option pricing models,
#              including the Black-Scholes Model, the Heston Model (Stochastic Volatility),
#              and the Merton Jump Diffusion Model. Additionally, it provides a heatmap
#              visualization of option prices using the Merton Jump Diffusion Model.
# -----------------------------------------------------------------------------------


# Black-Scholes Model
def black_scholes(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - X * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price


# Heston Model (Stochastic Volatility)
def heston(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0, num_simulations=10000, num_steps=100):
    dt = T / num_steps
    option_payoffs = []
    for _ in range(num_simulations):
        S_t = S_0
        v_t = v_0
        for _ in range(num_steps):
            z1, z2 = np.random.normal(size=2)
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
            S_t += r * S_t * dt + np.sqrt(v_t) * S_t * z1 * np.sqrt(dt)
            v_t += kappa * (theta - v_t) * dt + sigma_v * np.sqrt(v_t) * z2 * np.sqrt(dt)
            v_t = max(v_t, 0)
        option_payoff = max(S_t - X, 0)
        option_payoffs.append(option_payoff)
    average_payoff = np.mean(option_payoffs)
    option_price = np.exp(-r * T) * average_payoff
    return option_price


# Merton Jump Diffusion Model
def merton_jump_diffusion(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0, lambda_jump, m_jump, delta_jump, num_simulations=10000, num_steps=100):
    dt = T / num_steps
    S_t = np.full(num_simulations, S_0, dtype=np.float64)
    v_t = np.full(num_simulations, v_0, dtype=np.float64)
    for _ in range(num_steps):
        z1 = np.random.normal(size=num_simulations)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=num_simulations)
        S_t += r * S_t * dt + np.sqrt(v_t) * S_t * z1 * np.sqrt(dt)
        v_t += kappa * (theta - v_t) * dt + sigma_v * np.sqrt(v_t) * z2 * np.sqrt(dt)
        v_t = np.maximum(v_t, 0)
        jumps = np.random.rand(num_simulations) < lambda_jump * dt
        jump_sizes = np.random.normal(loc=m_jump, scale=delta_jump, size=num_simulations)
        S_t[jumps] *= np.exp(jump_sizes[jumps])
    option_payoffs = np.maximum(S_t - X, 0)
    average_payoff = np.mean(option_payoffs)
    option_price = np.exp(-r * T) * average_payoff
    return option_price


# Visualization (Heatmap)
def create_heatmap():
    # Define a grid of values for volatility and interest rates
    volatility_grid = np.linspace(0.1, 0.5, 10)  # From 10% to 50%
    interest_rate_grid = np.linspace(0.01, 0.1, 10)  # From 1% to 10%

    # Initialize a matrix to store results
    option_prices_matrix = np.zeros((len(interest_rate_grid), len(volatility_grid)))

    # Calculate option prices for each combination of volatility and interest rate
    for i, r in enumerate(interest_rate_grid):
        for j, sigma_v in enumerate(volatility_grid):
            price = merton_jump_diffusion(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0, lambda_jump, m_jump, delta_jump)
            option_prices_matrix[i, j] = price

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(option_prices_matrix, cmap='viridis', extent=[volatility_grid[0], volatility_grid[-1], interest_rate_grid[0], interest_rate_grid[-1]], aspect='auto', origin='lower')
    plt.colorbar(label='Option Price')
    plt.title('Option Prices Heatmap')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Interest Rate (%)')

    # Format axes to show percentages
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.show()


# Black-Scholes model parameters
S_0 = 100    # Current stock price
X = 100      # Strike price
T = 1        # Time to expiration in years
r = 0.05     # Risk-free interest rate
sigma = 0.2  # Volatility

# Heston model parameters
kappa = 3.0    # Mean reversion rate
theta = 0.04   # Long-term mean of volatility
sigma_v = 0.1  # Volatility of volatility
rho = -0.7     # Correlation between the asset return and volatility
v_0 = 0.04     # Initial variance

# Merton jump diffusion parameters
lambda_jump = 0.5  # Jump intensity
m_jump = -0.05     # Mean jump size
delta_jump = 0.1   # Jump-size volatility

# Calculate option prices
bs_price = black_scholes(S_0, X, T, r, sigma)
heston_price = heston(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0)
merton_jump_price = merton_jump_diffusion(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0, lambda_jump, m_jump, delta_jump)

# Print results
print("Black-Scholes Price:", bs_price)
print("Heston Model Price:", heston_price)
print("Merton Jump Diffusion Price:", merton_jump_price)

# Create heatmap visualization
create_heatmap()
