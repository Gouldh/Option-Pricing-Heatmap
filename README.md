# Option Pricing Models with Visualization

## Project Overview
This repository contains a financial analysis tool designed to calculate option prices using advanced pricing models such as Black-Scholes, Heston, and Merton Jump Diffusion. The tool is implemented in Python and is capable of handling complex mathematical and statistical functions, thanks to libraries like `numpy` and `scipy`.

## Features
- **Black-Scholes Model**: Utilizes the well-known Black-Scholes formula for pricing options.
- **Heston Model**: Incorporates stochastic volatility into the pricing model, capturing more complex market behaviors.
- **Merton Jump Diffusion Model**: Accounts for sudden price jumps in asset prices, refining the prediction of option prices.
- **Heatmap Visualization**: Generates a heatmap to visualize option prices across different volatilities and interest rates, offering intuitive insights into the data.

## Installation
To use this tool, clone the repository and set up a Python environment with the necessary dependencies.
```bash
git clone https://github.com/Gouldh/Option-Pricing-Heatmap.git
cd Option-Pricing-Heatmap
pip install -r requirements.txt
```

## Usage
After installation, you can run the script `main.py` to perform the pricing calculations and generate visualizations.

```bash
python main.py
```

Please ensure you have a compatible environment set up with all the necessary libraries, such as `numpy`, `scipy`, and `matplotlib`.

## Contributing
Contributions to this project are welcome. You can contribute in several ways:

1. Submitting a pull request for new features or bug fixes.
2. Improving documentation or examples.
3. Reporting issues or suggesting enhancements.

Please read `CONTRIBUTING.md` for details on the code of conduct, and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
