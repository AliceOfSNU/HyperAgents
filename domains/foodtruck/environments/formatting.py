from __future__ import annotations

from typing import Dict, List, Optional


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _format_money(value: float) -> str:
    return f"${value:.2f}"


def format_observation(
    day_idx: int,
    horizon_days: int,
    day_of_week: int,
    stage: str,
    funds: float,
    prep_ops_left: int,
    last_message: Optional[str],
    checked: Dict[str, bool],
    ingredients: List[Dict[str, str]],
    inventory: List[int],
    prices: List[float],
    recipes: List[Dict],
    menu: List[Optional[int]],
    menu_prices: List[float],
    summary: Optional[Dict[str, float]] = None,
) -> str:
    lines: List[str] = []
    dow_name = WEEKDAYS[day_of_week]
    lines.append(
        f"Day {day_idx + 1}/{horizon_days} ({dow_name}) | Stage: {stage} | "
        f"Funds: {_format_money(funds)} | Prep ops left: {prep_ops_left}"
    )
    if last_message:
        lines.append(f"Last action: {last_message}")

    if summary:
        lines.append("Daily summary:")
        lines.append(
            f"- Orders filled: {int(summary['orders_filled'])} | "
            f"Orders unfilled: {int(summary['orders_unfilled'])}"
        )
        breakdown = summary.get("order_breakdown")
        if breakdown:
            for recipe_name, counts in breakdown.items():
                lines.append(
                    f"- {recipe_name}: filled {int(counts['filled'])} | "
                    f"unfilled {int(counts['unfilled'])}"
                )
        lines.append(
            f"- Revenue: {_format_money(summary['revenue'])} | "
            f"Rent: {_format_money(summary['rent'])} | "
            f"Net: {_format_money(summary['net'])} | "
            f"Funds: {_format_money(summary['funds_end'])}"
        )

    if checked.get("storage", False):
        lines.append("Storage:")
        fridge_items = []
        pantry_items = []
        for idx, ing in enumerate(ingredients):
            qty = inventory[idx]
            if ing["storage"] == "fridge":
                fridge_items.append(f"{ing['name']}: {qty}")
            else:
                pantry_items.append(f"{ing['name']}: {qty}")
        lines.append(f"- Fridge: {', '.join(fridge_items)}")
        lines.append(f"- Pantry: {', '.join(pantry_items)}")

    if checked.get("market", False):
        lines.append("Market prices:")
        lines.append(
            ", ".join(f"{ing['name']}: {_format_money(prices[i])}" for i, ing in enumerate(ingredients))
        )

    if checked.get("recipes", False):
        lines.append("Recipes:")
        for idx, recipe in enumerate(recipes):
            ing_parts = ", ".join(
                f"{name} x{qty}" for name, qty in recipe["ingredients"].items()
            )
            lines.append(f"- {idx + 1}. {recipe['name']}: {ing_parts}")

    if checked.get("menu", False):
        lines.append("Menu:")
        for idx, recipe_id in enumerate(menu):
            if recipe_id is None:
                lines.append(f"- Slot {idx + 1}: empty")
            else:
                recipe_name = recipes[recipe_id]["name"]
                lines.append(
                    f"- Slot {idx + 1}: {recipe_name} @ {_format_money(menu_prices[idx])}"
                )

    return "\n".join(lines)
