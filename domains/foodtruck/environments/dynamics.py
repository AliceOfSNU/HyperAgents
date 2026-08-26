from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_BASE_PRICES: Dict[str, float] = {
    "tomato": 4.0,
    "lettuce": 3.5,
    "cheese": 2.5,
    "milk": 2.0,
    "chicken": 5.5,
    "beef": 6.0,
    "rice": 1.5,
    "beans": 1.2,
    "tortilla": 1.0,
    "pasta": 1.3,
}


def _deterministic_price(name: str, price_min: float, price_max: float) -> float:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return price_min + value * (price_max - price_min)


def init_prices(
    ingredient_names: List[str],
    price_min: float,
    price_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    base_prices: List[float] = []
    for name in ingredient_names:
        key = name.strip().lower()
        if key in DEFAULT_BASE_PRICES:
            price = DEFAULT_BASE_PRICES[key]
        else:
            price = _deterministic_price(key, price_min, price_max)
        base_prices.append(float(np.clip(price, price_min, price_max)))
    base_prices_arr = np.array(base_prices, dtype=float)
    prices = base_prices_arr.copy()
    return prices, base_prices_arr


def update_prices(
    rng: np.random.Generator,
    prices: np.ndarray,
    base_prices: np.ndarray,
    mean_reversion: float,
    volatility: float,
    price_min: float,
    price_max: float,
) -> np.ndarray:
    noise = rng.normal(0.0, volatility, size=prices.shape)
    next_prices = prices + mean_reversion * (base_prices - prices) + noise
    return np.clip(next_prices, price_min, price_max)


def init_preferences(
    rng: np.random.Generator,
    recipe_costs: np.ndarray,
    base_demand_range: Tuple[float, float],
    price_elasticity_range: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    min_demand, max_demand = base_demand_range
    if recipe_costs.size == 0:
        base_demands = np.array([], dtype=float)
    else:
        min_cost = float(recipe_costs.min())
        max_cost = float(recipe_costs.max())
        if max_cost == min_cost:
            cost_scaled = np.full(recipe_costs.shape, (min_demand + max_demand) / 2.0)
        else:
            cost_scaled = min_demand + (recipe_costs - min_cost) / (max_cost - min_cost) * (
                max_demand - min_demand
            )
        noise = rng.normal(0.0, 0.15 * (max_demand - min_demand), size=recipe_costs.shape)
        base_demands = np.clip(cost_scaled + noise, min_demand, max_demand)
    price_elasticities = rng.uniform(
        price_elasticity_range[0], price_elasticity_range[1], size=recipe_costs.shape
    )
    return {"base_demands": base_demands, "price_elasticities": price_elasticities}


def expected_demand(
    recipe_id: int,
    price: float,
    day_of_week: int,
    preferences: Dict[str, np.ndarray],
    weekday_multipliers: Tuple[float, ...],
) -> float:
    base = preferences["base_demands"][recipe_id]
    elasticity = preferences["price_elasticities"][recipe_id]
    multiplier = weekday_multipliers[day_of_week]
    return max(0.0, base * multiplier * np.exp(-elasticity * price))


def sample_orders(rng: np.random.Generator, expected: float) -> int:
    if expected <= 0:
        return 0
    return int(rng.poisson(expected))
