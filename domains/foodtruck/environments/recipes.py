from __future__ import annotations

from typing import Dict, List


DEFAULT_INGREDIENTS: List[Dict[str, str]] = [
    {"name": "tomato", "storage": "fridge"},
    {"name": "lettuce", "storage": "fridge"},
    {"name": "cheese", "storage": "fridge"},
    {"name": "milk", "storage": "fridge"},
    {"name": "chicken", "storage": "fridge"},
    {"name": "beef", "storage": "fridge"},
    {"name": "rice", "storage": "pantry"},
    {"name": "beans", "storage": "pantry"},
    {"name": "tortilla", "storage": "pantry"},
    {"name": "pasta", "storage": "pantry"},
]

DEFAULT_RECIPES: List[Dict[str, Dict[str, int]]] = [
    {"name": "veggie_wrap", "ingredients": {"tomato": 1, "lettuce": 1, "tortilla": 1}},
    {"name": "chicken_bowl", "ingredients": {"chicken": 1, "rice": 1, "beans": 1}},
    {"name": "beef_tacos", "ingredients": {"beef": 1, "tortilla": 2, "lettuce": 1}},
    {"name": "cheese_pasta", "ingredients": {"pasta": 1, "cheese": 1, "milk": 1}},
    {"name": "tomato_pasta", "ingredients": {"pasta": 1, "tomato": 2}},
    {"name": "rice_and_beans", "ingredients": {"rice": 1, "beans": 2}},
    {"name": "chicken_wrap", "ingredients": {"chicken": 1, "tortilla": 1, "lettuce": 1}},
    {"name": "beef_bowl", "ingredients": {"beef": 1, "rice": 1, "tomato": 1}},
    {"name": "cheese_wrap", "ingredients": {"cheese": 1, "tortilla": 1, "lettuce": 1}},
    {"name": "tomato_salad", "ingredients": {"tomato": 2, "lettuce": 2}},
]


def build_ingredient_catalog(overrides: List[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
    ingredients = overrides if overrides else DEFAULT_INGREDIENTS
    names = [item["name"] for item in ingredients]
    if len(names) != len(set(names)):
        raise ValueError("Ingredient names must be unique")
    for item in ingredients:
        if item["storage"] not in {"fridge", "pantry"}:
            raise ValueError(f"Invalid storage type for ingredient: {item}")
    return ingredients


def build_recipe_book(
    num_recipes: int,
    ingredient_catalog: List[Dict[str, str]],
    overrides: List[Dict[str, Dict[str, int]]] | None = None,
) -> List[Dict[str, Dict[str, int]]]:
    recipes = overrides if overrides else DEFAULT_RECIPES
    if num_recipes > len(recipes):
        raise ValueError("num_recipes exceeds available recipes")
    names = [recipe["name"] for recipe in recipes]
    if len(names) != len(set(names)):
        raise ValueError("Recipe names must be unique")
    ingredient_names = {item["name"] for item in ingredient_catalog}
    for recipe in recipes[:num_recipes]:
        for ing_name, qty in recipe["ingredients"].items():
            if ing_name not in ingredient_names:
                raise ValueError(f"Recipe uses unknown ingredient: {ing_name}")
            if qty <= 0:
                raise ValueError(f"Ingredient quantity must be positive: {recipe}")
    return recipes[:num_recipes]
