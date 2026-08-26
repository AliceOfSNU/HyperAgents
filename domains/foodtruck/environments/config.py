# Vendored from CMU 11-766 HW2's FoodTruckEnv (11-766-hw2/release/11766_hw2_p1_2/),
# unchanged except this note -- see domains/foodtruck/eval.py's own module
# docstring for how it's wired into this project's harness.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class EnvConfig:
    # Episode
    horizon_days: int = 20
    start_funds: float = 500.0
    daily_rent: float = 2.0
    bankrupt_terminates: bool = False

    # Capacities
    fridge_capacity: int = 20
    pantry_capacity: int = 20

    # Recipes/menu
    num_recipes: int = 10
    max_menu_items: int = 4
    default_menu_price: float = 8.0

    # Prep stage
    max_prep_ops_per_day: int = 20
    invalid_action_costs_prep: bool = True

    # Market pricing
    price_min: float = 0.5
    price_max: float = 20.0
    price_mean_reversion: float = 0.2
    price_volatility: float = 0.5

    # Demand (hidden)
    base_demand_range: Tuple[float, float] = (8.0, 24.0)
    price_elasticity_range: Tuple[float, float] = (0.05, 0.18)
    weekday_multipliers: Tuple[float, ...] = (1.0, 1.0, 1.05, 1.05, 1.1, 1.2, 1.15)

    # Action limits
    max_buy_qty: int = 10
    max_trash_qty: int = 10
    price_step: float = 0.5

    # Text I/O
    max_observation_chars: int = 8000
    max_action_chars: int = 200

    # Ingredients/recipes overrides (optional)
    ingredient_overrides: List[Dict[str, str]] = field(default_factory=list)
    recipe_overrides: List[Dict[str, Dict[str, int]]] = field(default_factory=list)

    def validate(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.fridge_capacity <= 0 or self.pantry_capacity <= 0:
            raise ValueError("storage capacities must be positive")
        if self.num_recipes <= 0:
            raise ValueError("num_recipes must be positive")
        if self.max_menu_items <= 0:
            raise ValueError("max_menu_items must be positive")
        if self.max_prep_ops_per_day <= 0:
            raise ValueError("max_prep_ops_per_day must be positive")
        if self.price_min <= 0 or self.price_max <= self.price_min:
            raise ValueError("invalid price_min/price_max range")
        if self.price_step <= 0:
            raise ValueError("price_step must be positive")
        if len(self.weekday_multipliers) != 7:
            raise ValueError("weekday_multipliers must have length 7")
