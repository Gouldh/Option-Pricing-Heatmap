# Option Pricing Models with Visualization

## Project Overview
This repository offers a financial analysis tool that computes option prices using sophisticated models like Black-Scholes, Heston, and Merton Jump Diffusion. It's written in Python and leverages libraries such as `numpy` and `scipy` for complex calculations.

## Features
- **Black-Scholes Model**: Applies the renowned Black-Scholes formula for option pricing.
- **Heston Model**: Includes stochastic volatility in the pricing model to capture intricate market dynamics.
- **Merton Jump Diffusion Model**: Considers abrupt price changes in asset valuation for more accurate option pricing.
- **Heatmap Visualization**: Creates heatmaps to illustrate option price variations across different volatilities and interest rates, providing clear visual data interpretation.

## Installation
Follow these steps to set up the tool:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gouldh/Option-Pricing-Heatmap.git
   ```
2. Navigate to the repository's directory:
   ```bash
   cd Option-Pricing-Heatmap
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
To run the pricing models and generate visualizations, execute the `main.py` script:

```bash
python main.py
```

Make sure your Python environment includes all necessary libraries, such as `numpy`, `scipy`, and `matplotlib`, for the tool to function properly.

## License
This project is open-sourced under the MIT License. For more information, refer to the `LICENSE` file.
