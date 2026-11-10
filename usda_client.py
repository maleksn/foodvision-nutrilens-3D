import os
import urllib.request
import urllib.parse
import json

# Fallback scientific food densities (g / cm^3) accounting for food state and porosity
FOOD_DENSITY_TABLE = {
    "chicken": 0.95,
    "cooked chicken": 0.95,
    "shredded cooked chicken": 0.92,
    "grilled chicken": 0.95,
    "french fries": 0.52,   # Fried potatoes with porosity/air gaps
    "fries": 0.52,
    "potato": 0.75,
    "rice": 0.82,           # Cooked white/brown rice packed
    "cooked rice": 0.82,
    "broccoli": 0.40,       # Florets with significant air gaps
    "steamed broccoli": 0.42,
    "chickpeas": 0.88,      # Cooked chickpeas
    "green beans": 0.55,    # Whole green beans with stacking voids
    "hummus": 1.05,         # Dense paste/puree
    "sauce": 1.02,
    "beef": 1.02,
    "meat": 1.02,
    "fish": 0.90,
    "salad": 0.25,
    "bread": 0.35,
    "egg": 0.98,
    "default": 0.85
}

def load_api_key():
    key = os.environ.get("USDA_API_KEY")
    if key:
        return key.strip().strip('"').strip("'")
    
    # Try reading from .env in current or parent dirs
    for path in [".env", "../.env", "/mnt/d/NutriLens_Backend/.env"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("USDA_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
    return None

def get_food_density(food_name: str) -> float:
    name_lower = food_name.lower().strip()
    # Check exact match
    if name_lower in FOOD_DENSITY_TABLE:
        return FOOD_DENSITY_TABLE[name_lower]
    # Check partial match
    for k, v in FOOD_DENSITY_TABLE.items():
        if k in name_lower or name_lower in k:
            return v
    return FOOD_DENSITY_TABLE["default"]

STANDARD_NUTRITION_PROFILES = {
    "chicken": {"calories": 165.0, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "desc": "Chicken breast, cooked, skinless"},
    "cooked chicken": {"calories": 165.0, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "desc": "Chicken breast, cooked, skinless"},
    "cooked chicken breast": {"calories": 165.0, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "desc": "Chicken breast, cooked, skinless"},
    "chicken breast": {"calories": 165.0, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "desc": "Chicken breast, cooked, skinless"},
    "shredded cooked chicken": {"calories": 165.0, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "desc": "Chicken breast, shredded cooked"},
    "french fries": {"calories": 196.0, "protein": 2.5, "carbs": 25.0, "fat": 9.5, "desc": "French fries, cooked"},
    "fries": {"calories": 196.0, "protein": 2.5, "carbs": 25.0, "fat": 9.5, "desc": "French fries, cooked"},
    "white rice": {"calories": 130.0, "protein": 2.7, "carbs": 28.2, "fat": 0.3, "desc": "White rice, cooked"},
    "rice": {"calories": 130.0, "protein": 2.7, "carbs": 28.2, "fat": 0.3, "desc": "White rice, cooked"},
    "cooked rice": {"calories": 130.0, "protein": 2.7, "carbs": 28.2, "fat": 0.3, "desc": "White rice, cooked"},
    "broccoli": {"calories": 35.0, "protein": 2.4, "carbs": 7.2, "fat": 0.4, "desc": "Broccoli, steamed"},
    "steamed broccoli": {"calories": 35.0, "protein": 2.4, "carbs": 7.2, "fat": 0.4, "desc": "Broccoli, steamed"},
    "beef": {"calories": 250.0, "protein": 26.0, "carbs": 0.0, "fat": 15.0, "desc": "Beef steak, cooked"},
    "meat": {"calories": 250.0, "protein": 26.0, "carbs": 0.0, "fat": 15.0, "desc": "Meat, cooked"},
    "fish": {"calories": 170.0, "protein": 22.0, "carbs": 0.0, "fat": 8.0, "desc": "Fish fillet, cooked"},
    "salmon": {"calories": 208.0, "protein": 20.0, "carbs": 0.0, "fat": 13.0, "desc": "Salmon fillet, cooked"},
    "pasta": {"calories": 131.0, "protein": 5.2, "carbs": 25.0, "fat": 1.1, "desc": "Pasta, cooked"},
    "potatoes": {"calories": 87.0, "protein": 1.9, "carbs": 20.0, "fat": 0.1, "desc": "Potatoes, boiled"}
}

def query_usda_nutrients(food_name: str):
    api_key = load_api_key()
    density = get_food_density(food_name)
    fname_clean = food_name.lower().strip()
    
    # Check baseline fallback profile
    base_prof = None
    if fname_clean in STANDARD_NUTRITION_PROFILES:
        base_prof = STANDARD_NUTRITION_PROFILES[fname_clean]
    else:
        for k, v in STANDARD_NUTRITION_PROFILES.items():
            if k in fname_clean or fname_clean in k:
                base_prof = v
                break

    result = {
        "query": food_name,
        "density_g_cm3": density,
        "calories_per_100g": base_prof["calories"] if base_prof else 150.0,
        "protein_per_100g": base_prof["protein"] if base_prof else 10.0,
        "carbs_per_100g": base_prof["carbs"] if base_prof else 15.0,
        "fat_per_100g": base_prof["fat"] if base_prof else 5.0,
        "description": base_prof["desc"] if base_prof else food_name,
        "source": "Standard Nutritional Reference"
    }

    if not api_key:
        return result

    # Find the core food noun (e.g. 'broccoli', 'chicken', 'rice', 'fries')
    common_nouns = ["chicken", "rice", "broccoli", "fries", "beef", "meat", "salmon", "fish", "potato", "pasta"]
    core_noun = None
    for cn in common_nouns:
        if cn in fname_clean:
            core_noun = cn
            break

    search_term = food_name if not core_noun else f"{core_noun} {fname_clean}".strip()
    encoded_query = urllib.parse.quote(search_term)
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={encoded_query}&pageSize=5&api_key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "NutriLens-3D/1.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            foods = data.get("foods", [])
            matched_food = None
            if foods:
                # Find the first food item whose description actually contains the core noun
                if core_noun:
                    for f in foods:
                        if core_noun in f.get("description", "").lower():
                            matched_food = f
                            break
                if matched_food is None:
                    matched_food = foods[0]
                    
                result["description"] = matched_food.get("description", food_name)
                result["source"] = "USDA FoodData Central"
                for n in matched_food.get("foodNutrients", []):
                    name = n.get("nutrientName", "").lower()
                    val = float(n.get("value", 0.0))
                    if "energy" in name and "kcal" in n.get("unitName", "").lower():
                        result["calories_per_100g"] = val
                    elif "protein" in name:
                        result["protein_per_100g"] = val
                    elif "carbohydrate" in name:
                        result["carbs_per_100g"] = val
                    elif "total lipid" in name or "fat" in name:
                        result["fat_per_100g"] = val
    except Exception as e:
        print(f"Warning: USDA API query for '{food_name}' failed ({e}). Using sports nutrition profile.")
        
    return result

def calculate_macros(food_name: str, weight_g: float):
    info = query_usda_nutrients(food_name)
    factor = weight_g / 100.0
    return {
        "food": food_name,
        "matched_desc": info["description"],
        "weight_g": round(weight_g, 1),
        "density_g_cm3": info["density_g_cm3"],
        "calories": round(info["calories_per_100g"] * factor, 1),
        "protein_g": round(info["protein_per_100g"] * factor, 1),
        "carbs_g": round(info["carbs_per_100g"] * factor, 1),
        "fat_g": round(info["fat_per_100g"] * factor, 1),
        "source": info["source"]
    }
