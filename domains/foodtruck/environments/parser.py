from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ParsedCommand:
    op: str
    args: Tuple
    error: Optional[str] = None


def _parse_index(token: str, max_value: int, label: str) -> Tuple[Optional[int], Optional[str]]:
    if not token.isdigit():
        return None, f"{label} must be an integer"
    value = int(token)
    if value < 1 or value > max_value:
        return None, f"{label} must be between 1 and {max_value}"
    return value - 1, None


def _parse_qty(token: str, max_qty: int) -> Tuple[Optional[int], Optional[str]]:
    if not token.isdigit():
        return None, "quantity must be an integer"
    qty = int(token)
    if qty <= 0 or qty > max_qty:
        return None, f"quantity must be between 1 and {max_qty}"
    return qty, None


def _parse_name_or_id(token: str, names: List[str], label: str) -> Tuple[Optional[int], Optional[str]]:
    if token.isdigit():
        idx, err = _parse_index(token, len(names), label)
        return idx, err
    token = token.strip().lower()
    if token not in names:
        return None, f"unknown {label}: {token}"
    return names.index(token), None


def parse_command(
    text: str,
    ingredient_names: List[str],
    recipe_names: List[str],
    max_menu_items: int,
) -> ParsedCommand:
    cleaned = text.strip().lower()
    if not cleaned:
        return ParsedCommand(op="invalid", args=(), error="empty command")
    tokens = cleaned.split()

    if tokens[0] == "check" and len(tokens) == 2:
        if tokens[1] in {"storage", "market", "recipes", "menu"}:
            return ParsedCommand(op="check", args=(tokens[1],))
        return ParsedCommand(op="invalid", args=(), error="unknown check target")

    if tokens[0] == "buy" and len(tokens) == 3:
        ing_id, err = _parse_name_or_id(tokens[1], ingredient_names, "ingredient")
        if err:
            return ParsedCommand(op="invalid", args=(), error=err)
        qty, err = _parse_qty(tokens[2], max_qty=9999)
        if err:
            return ParsedCommand(op="invalid", args=(), error=err)
        return ParsedCommand(op="buy", args=(ing_id, qty))

    if tokens[0] == "trash" and len(tokens) == 3:
        ing_id, err = _parse_name_or_id(tokens[1], ingredient_names, "ingredient")
        if err:
            return ParsedCommand(op="invalid", args=(), error=err)
        qty, err = _parse_qty(tokens[2], max_qty=9999)
        if err:
            return ParsedCommand(op="invalid", args=(), error=err)
        return ParsedCommand(op="trash", args=(ing_id, qty))

    if tokens[0] == "set" and len(tokens) == 5 and tokens[1] == "menu":
        slot_id, err = _parse_index(tokens[2], max_menu_items, "slot")
        if err:
            return ParsedCommand(op="invalid", args=(), error=err)
        recipe_id, err = _parse_name_or_id(tokens[3], recipe_names, "recipe")
        if err:
            return ParsedCommand(op="invalid", args=(), error=err)
        try:
            price = float(tokens[4])
        except ValueError:
            return ParsedCommand(op="invalid", args=(), error="price must be a number")
        return ParsedCommand(op="set_menu", args=(slot_id, recipe_id, price))

    if tokens[0] == "clear" and len(tokens) == 3 and tokens[1] == "menu":
        slot_id, err = _parse_index(tokens[2], max_menu_items, "slot")
        if err:
            return ParsedCommand(op="invalid", args=(), error=err)
        return ParsedCommand(op="clear_menu", args=(slot_id,))

    if cleaned in {"end prep", "endprep"}:
        return ParsedCommand(op="end_prep", args=())

    return ParsedCommand(op="invalid", args=(), error="unrecognized command")
