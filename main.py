import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# -----------------------------------------------------------------------------------
# Author: Hunter Gould
# Date: 11/24/2023
# Description: This project involves the implementation of various option pricing models,
#              including the Black-Scholes Model, the Heston Model, and the Merton Jump
#              Diffusion Model. Additionally, it provides a heatmap visualization of
#              option prices using the Merton Jump Diffusion Model.
# -----------------------------------------------------------------------------------


# Black-Scholes Model
def black_scholes(S, X, T, r, sigma):
    """
    Calculates the Black-Scholes option price.
    :param float S: Current price of the underlying asset.
    :param float X: Strike price of the option.
    :param float T: Time to expiration in years.
    :param float r: Risk-free interest rate.
    :param float sigma: Volatility of the underlying asset.
    :return: The calculated call option price as a float.
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - X * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price


# Heston Model (Stochastic Volatility)
def heston(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0, num_simulations=10000, num_steps=100):
    """
    Calculates the option price using the Heston model.
    :param float S_0: Initial price of the underlying asset.
    :param float X: Strike price of the option.
    :param float T: Time to expiration in years.
    :param float r: Risk-free interest rate.
    :param float kappa: Mean reversion rate of volatility.
    :param float theta: Long-term mean of volatility.
    :param float sigma_v: Volatility of volatility.
    :param float rho: Correlation between asset return and volatility.
    :param float v_0: Initial variance.
    :param int num_simulations: Number of simulations. Default is 10000.
    :param int num_steps: Number of steps in the simulation. Default is 100.
    :return: The estimated option price using the Heston model as a float.
    """
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
def merton_jump_diffusion(S_0, X, T, r, sigma, lambda_jump, m_jump, delta_jump, num_simulations=10000, num_steps=100):
    """
    Calculates the option price using the Merton Jump Diffusion model.
    :param float S_0: Initial price of the underlying asset.
    :param float X: Strike price of the option.
    :param float T: Time to expiration in years.
    :param float r: Risk-free interest rate.
    :param float sigma: Volatility of the underlying asset.
    :param float lambda_jump: Intensity of the jumps.
    :param float m_jump: Mean of the jump size.
    :param float delta_jump: Volatility of the jump size.
    :param int num_simulations: Number of simulations. Default is 10000.
    :param int num_steps: Number of steps in the simulation. Default is 100.
    :return: The estimated option price using the Merton Jump Diffusion model as a float.
    """
    dt = T / num_steps
    S_t = np.full(num_simulations, S_0, dtype=np.float64)

    for _ in range(num_steps):
        z = np.random.normal(size=num_simulations)
        jump_sizes = np.random.normal(loc=m_jump, scale=delta_jump, size=num_simulations)
        jumps = np.random.poisson(lambda_jump * dt, size=num_simulations)
        S_t *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        S_t *= np.exp(jumps * jump_sizes)

    option_payoffs = np.maximum(S_t - X, 0)
    average_payoff = np.mean(option_payoffs)
    option_price = np.exp(-r * T) * average_payoff

    return option_price


# Visualization (Heatmap)
def create_heatmap(S_0, X, T, lambda_jump, m_jump, delta_jump,
                   volatility_range=(0.1, 0.5), interest_rate_range=(0.01, 0.1),
                   volatility_steps=12, interest_rate_steps=12):
    """
    Creates a heatmap visualization of option prices using the Merton Jump Diffusion model.
    :param float S_0: Initial price of the underlying asset.
    :param float X: Strike price of the option.
    :param float T: Time to expiration in years.
    :param float lambda_jump: Intensity of the jumps.
    :param float m_jump: Mean of the jump size.
    :param float delta_jump: Volatility of the jump size.
    :param tuple volatility_range: Range of volatilities to iterate over. Default is (0.1, 0.5).
    :param tuple interest_rate_range: Range of interest rates to iterate over. Default is (0.01, 0.1).
    :param int volatility_steps: Number of steps in the volatility grid. Default is 12.
    :param int interest_rate_steps: Number of steps in the interest rate grid. Default is 12.
    :return: None
    """

    # Define a grid of values for volatility and interest rates
    volatility_grid = np.linspace(volatility_range[0], volatility_range[1], volatility_steps)
    interest_rate_grid = np.linspace(interest_rate_range[0], interest_rate_range[1], interest_rate_steps)

    # Initialize a matrix to store results
    option_prices_matrix = np.zeros((len(interest_rate_grid), len(volatility_grid)))

    # Calculate option prices for each combination of volatility and interest rate
    for i, r in enumerate(interest_rate_grid):
        for j, sigma_v in enumerate(volatility_grid):
            price = merton_jump_diffusion(S_0, X, T, r, sigma, lambda_jump, m_jump, delta_jump)
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


if __name__ == "__main__":
    # Black-Scholes model parameters
    S_0 = 100     # Current stock price
    X = 110       # Strike price
    T = 1         # Time to expiration in years
    r = 0.03      # Risk-free interest rate
    sigma = 0.25  # Volatility

    # Heston model parameters
    kappa = 2.0    # Mean reversion rate
    theta = 0.06   # Long-term mean of volatility
    sigma_v = 0.2  # Volatility of volatility
    rho = -0.5     # Correlation between the asset return and volatility
    v_0 = 0.05     # Initial variance

    # Merton jump diffusion parameters
    lambda_jump = 0.75  # Jump intensity
    m_jump = -0.06      # Mean jump size
    delta_jump = 0.12   # Jump-size volatility

    # Calculate option prices
    bs_price = black_scholes(S_0, X, T, r, sigma)
    heston_price = heston(S_0, X, T, r, kappa, theta, sigma_v, rho, v_0)
    merton_jump_price = merton_jump_diffusion(S_0, X, T, r, sigma, lambda_jump, m_jump, delta_jump)

    # Print results
    print(f"Black-Scholes Price: ${bs_price:.2f}")
    print(f"Heston Model Price: ${heston_price:.2f}")
    print(f"Merton Jump Diffusion Price: ${merton_jump_price:.2f}")

    # Create heatmap visualization
    create_heatmap(S_0, X, T, lambda_jump, m_jump, delta_jump)
