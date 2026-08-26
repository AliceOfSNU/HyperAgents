# Vendored from CMU 11-766 HW2's FoodTruckEnv, adapted only to use relative
# imports (package layout, not a bare-directory script) -- see
# domains/foodtruck/eval.py's own module docstring for how it's wired in.
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .config import EnvConfig
from .dynamics import expected_demand, init_preferences, init_prices, sample_orders, update_prices
from .formatting import format_observation
from .parser import ParsedCommand, parse_command
from .recipes import build_ingredient_catalog, build_recipe_book


class FoodTruckEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.config.validate()

        self.ingredients = build_ingredient_catalog(self.config.ingredient_overrides)
        self.recipes = build_recipe_book(
            self.config.num_recipes, self.ingredients, self.config.recipe_overrides
        )

        self.ingredient_names = [item["name"] for item in self.ingredients]
        self.recipe_names = [item["name"] for item in self.recipes]

        self.action_space = gym.spaces.Text(max_length=self.config.max_action_chars)
        self.observation_space = gym.spaces.Text(max_length=self.config.max_observation_chars)

        self.rng: Optional[np.random.Generator] = None
        self.day_idx: int = 0
        self.funds: float = self.config.start_funds
        self.prep_ops_used: int = 0
        self.inventory: np.ndarray = np.zeros(len(self.ingredients), dtype=int)
        self.prices: np.ndarray = np.zeros(len(self.ingredients), dtype=float)
        self.base_prices: np.ndarray = np.zeros(len(self.ingredients), dtype=float)
        self.menu: List[Optional[int]] = [None for _ in range(self.config.max_menu_items)]
        self.menu_prices: List[float] = [
            self.config.default_menu_price for _ in range(self.config.max_menu_items)
        ]
        self.preferences: Dict[str, np.ndarray] = {}
        self.checked: Dict[str, bool] = {
            "storage": False,
            "market": False,
            "recipes": False,
            "menu": False,
        }
        self.last_message: Optional[str] = None
        self.last_summary: Optional[Dict[str, float]] = None

        self._terminated = False
        self._truncated = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.day_idx = 0
        self.funds = float(self.config.start_funds)
        self.prep_ops_used = 0
        self.inventory = np.zeros(len(self.ingredients), dtype=int)
        self.prices, self.base_prices = init_prices(
            self.ingredient_names,
            self.config.price_min,
            self.config.price_max,
        )
        self.menu = [None for _ in range(self.config.max_menu_items)]
        self.menu_prices = [
            self.config.default_menu_price for _ in range(self.config.max_menu_items)
        ]
        recipe_costs = np.array(
            [
                sum(
                    self.base_prices[self.ingredient_names.index(ing_name)] * qty
                    for ing_name, qty in recipe["ingredients"].items()
                )
                for recipe in self.recipes
            ],
            dtype=float,
        )
        self.preferences = init_preferences(
            self.rng,
            recipe_costs,
            self.config.base_demand_range,
            self.config.price_elasticity_range,
        )
        self.checked = {"storage": False, "market": False, "recipes": False, "menu": False}
        self.last_message = None
        self.last_summary = None
        self._terminated = False
        self._truncated = False

        obs = self._build_observation()
        info = {"day": self.day_idx}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode has terminated. Call reset() to start a new episode.")
        if self.rng is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        self.last_message = None
        self.last_summary = None

        parsed = parse_command(action, self.ingredient_names, self.recipe_names, self.config.max_menu_items)
        reward = 0.0
        opened = False

        if parsed.op == "invalid":
            self.last_message = f"Invalid command: {parsed.error}"
            if self.config.invalid_action_costs_prep:
                self.prep_ops_used += 1
        else:
            opened = self._apply_command(parsed)
            self.prep_ops_used += 1

        if not opened and self.prep_ops_used >= self.config.max_prep_ops_per_day:
            opened = True

        if opened:
            reward = self._run_open_stage()

        self._terminated = self.day_idx >= self.config.horizon_days
        if self.config.bankrupt_terminates and self.funds < 0:
            self._terminated = True

        obs = self._build_observation()
        for key in self.checked:
            self.checked[key] = False
        info = {"day": self.day_idx, "parsed_op": parsed.op}
        return obs, float(reward), self._terminated, self._truncated, info

    def render(self):
        return self._build_observation()

    def close(self):
        pass

    def _apply_command(self, parsed: ParsedCommand) -> bool:
        op = parsed.op
        if op == "check":
            target = parsed.args[0]
            if target in self.checked:
                self.checked[target] = True
                self.last_message = f"Checked {target}"
            else:
                self.last_message = "Invalid check target"
            return False

        if op == "buy":
            ing_id, qty = parsed.args
            return self._buy_ingredient(ing_id, qty)

        if op == "trash":
            ing_id, qty = parsed.args
            return self._trash_ingredient(ing_id, qty)

        if op == "set_menu":
            slot_id, recipe_id, price = parsed.args
            if recipe_id in self.menu and self.menu[slot_id] != recipe_id:
                self.last_message = "Recipe already on menu in another slot"
                return False
            ok, message = self._set_price(slot_id, price)
            if not ok:
                self.last_message = message
                return False
            self.menu[slot_id] = recipe_id
            self.last_message = (
                f"Set menu slot {slot_id + 1} to {self.recipe_names[recipe_id]} @ ${price:.2f}"
            )
            return False

        if op == "clear_menu":
            slot_id = parsed.args[0]
            self.menu[slot_id] = None
            self.last_message = f"Cleared menu slot {slot_id + 1}"
            return False

        if op == "end_prep":
            self.last_message = "Ended prep"
            return True

        self.last_message = "Unknown operation"
        return False

    def _set_price(self, slot_id: int, price: float) -> Tuple[bool, str]:
        if price < self.config.price_min or price > self.config.price_max:
            return False, "Price out of bounds"
        step = self.config.price_step
        if abs(round((price - self.config.price_min) / step) * step + self.config.price_min - price) > 1e-6:
            return False, "Price must align with price_step grid"
        self.menu_prices[slot_id] = float(price)
        return True, f"Set price for slot {slot_id + 1} to ${price:.2f}"

    def _buy_ingredient(self, ing_id: int, qty: int) -> bool:
        if qty > self.config.max_buy_qty:
            self.last_message = f"Buy qty exceeds max {self.config.max_buy_qty}"
            return False
        price = self.prices[ing_id]
        cost = float(price * qty)
        if self.funds < cost:
            self.last_message = "Insufficient funds"
            return False
        storage_type = self.ingredients[ing_id]["storage"]
        if not self._has_capacity(storage_type, qty):
            self.last_message = "Insufficient storage capacity"
            return False
        self.inventory[ing_id] += qty
        self.funds -= cost
        self.last_message = f"Bought {qty} {self.ingredient_names[ing_id]} for ${cost:.2f}"
        return False

    def _trash_ingredient(self, ing_id: int, qty: int) -> bool:
        if qty > self.config.max_trash_qty:
            self.last_message = f"Trash qty exceeds max {self.config.max_trash_qty}"
            return False
        if self.inventory[ing_id] < qty:
            self.last_message = "Not enough inventory to trash"
            return False
        self.inventory[ing_id] -= qty
        self.last_message = f"Trashed {qty} {self.ingredient_names[ing_id]}"
        return False

    def _has_capacity(self, storage_type: str, qty: int) -> bool:
        if storage_type == "fridge":
            current = sum(
                self.inventory[i]
                for i, ing in enumerate(self.ingredients)
                if ing["storage"] == "fridge"
            )
            return current + qty <= self.config.fridge_capacity
        current = sum(
            self.inventory[i]
            for i, ing in enumerate(self.ingredients)
            if ing["storage"] == "pantry"
        )
        return current + qty <= self.config.pantry_capacity

    def _run_open_stage(self) -> float:
        revenue = 0.0
        orders_filled = 0
        orders_unfilled = 0
        order_breakdown: Dict[str, Dict[str, int]] = {}

        day_of_week = self.day_idx % 7
        for slot_id, recipe_id in enumerate(self.menu):
            if recipe_id is None:
                continue
            price = self.menu_prices[slot_id]
            expected = expected_demand(
                recipe_id,
                price,
                day_of_week,
                self.preferences,
                self.config.weekday_multipliers,
            )
            orders = sample_orders(self.rng, expected)
            if orders <= 0:
                continue

            max_fill = self._max_fulfillable_orders(recipe_id)
            filled = min(orders, max_fill)
            if filled > 0:
                self._consume_ingredients(recipe_id, filled)
                revenue += filled * price
            orders_filled += filled
            orders_unfilled += orders - filled
            recipe_name = self.recipe_names[recipe_id]
            if recipe_name not in order_breakdown:
                order_breakdown[recipe_name] = {"filled": 0, "unfilled": 0}
            order_breakdown[recipe_name]["filled"] += int(filled)
            order_breakdown[recipe_name]["unfilled"] += int(orders - filled)

        rent = float(self.config.daily_rent)
        self.funds += revenue
        self.funds -= rent
        net = revenue - rent

        self.last_summary = {
            "orders_filled": float(orders_filled),
            "orders_unfilled": float(orders_unfilled),
            "order_breakdown": order_breakdown,
            "revenue": float(revenue),
            "rent": float(rent),
            "net": float(net),
            "funds_end": float(self.funds),
        }

        self.day_idx += 1
        if self.day_idx < self.config.horizon_days:
            self.prices = update_prices(
                self.rng,
                self.prices,
                self.base_prices,
                self.config.price_mean_reversion,
                self.config.price_volatility,
                self.config.price_min,
                self.config.price_max,
            )
            self.prep_ops_used = 0
            self.checked["storage"] = False
            self.checked["market"] = False
            self.checked["menu"] = False

        return float(net)

    def _max_fulfillable_orders(self, recipe_id: int) -> int:
        recipe = self.recipes[recipe_id]
        max_orders = float("inf")
        for ing_name, qty in recipe["ingredients"].items():
            ing_idx = self.ingredient_names.index(ing_name)
            max_orders = min(max_orders, self.inventory[ing_idx] // qty)
        return int(max_orders) if max_orders != float("inf") else 0

    def _consume_ingredients(self, recipe_id: int, count: int) -> None:
        recipe = self.recipes[recipe_id]
        for ing_name, qty in recipe["ingredients"].items():
            ing_idx = self.ingredient_names.index(ing_name)
            self.inventory[ing_idx] -= qty * count

    def _build_observation(self) -> str:
        prep_ops_left = max(self.config.max_prep_ops_per_day - self.prep_ops_used, 0)
        stage = "prep" if not self._terminated else "done"
        obs = format_observation(
            day_idx=min(self.day_idx, self.config.horizon_days - 1),
            horizon_days=self.config.horizon_days,
            day_of_week=self.day_idx % 7,
            stage=stage,
            funds=self.funds,
            prep_ops_left=prep_ops_left,
            last_message=self.last_message,
            checked=self.checked,
            ingredients=self.ingredients,
            inventory=self.inventory.tolist(),
            prices=self.prices.tolist(),
            recipes=self.recipes,
            menu=self.menu,
            menu_prices=self.menu_prices,
            summary=self.last_summary,
        )
        return obs
