#!/usr/bin/env python3
import json
import math
from collections import defaultdict
from pathlib import Path

BASE_ADDRESS = "Carrer Taronger 6, 46730 Gandia, Spain"
BASE_COORD = (38.997532, -0.1573372)

PLACES_PATH = Path("gandia_places_extracted.json")
PLANS_PATH = Path("day_trip_plans.json")

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_LABEL = {
    "N": "North",
    "NE": "Northeast",
    "E": "East",
    "SE": "Southeast",
    "S": "South",
    "SW": "Southwest",
    "W": "West",
    "NW": "Northwest",
}

MEAL_HINTS = [
    "restaurant",
    "mediterranean",
    "fast food",
    "authentic japanese",
    "tapas",
    "cured ham bar",
]
DRINK_HINTS = ["winery", "bakery", "pastries", "coffee"]
SHOP_HINTS = [
    "greengrocer",
    "supermarket",
    "market",
    "hypermarket",
    "department store",
    "general store",
    "discount supermarket",
    "butcher shop",
    "shopping mall",
    "gas station",
]


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def euclidean_distance_km(a_lat, a_lon, b_lat, b_lon):
    # Fast local-earth approximation for clustering/nearest-neighbor decisions.
    avg_lat = math.radians((a_lat + b_lat) / 2.0)
    dy = (b_lat - a_lat) * 110.574
    dx = (b_lon - a_lon) * 111.320 * math.cos(avg_lat)
    return math.sqrt(dx * dx + dy * dy)


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    r = 6371.0
    p1 = math.radians(a_lat)
    p2 = math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    x = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(x))


def bearing_deg(a_lat, a_lon, b_lat, b_lon):
    y = math.sin(math.radians(b_lon - a_lon)) * math.cos(math.radians(b_lat))
    x = math.cos(math.radians(a_lat)) * math.sin(math.radians(b_lat)) - math.sin(
        math.radians(a_lat)
    ) * math.cos(math.radians(b_lat)) * math.cos(math.radians(b_lon - a_lon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def sector_from_bearing(bearing):
    idx = int(((bearing + 22.5) % 360) // 45)
    return COMPASS[idx]


def quadrant_from_bearing(bearing):
    if bearing < 45 or bearing >= 315:
        return "N"
    if bearing < 135:
        return "E"
    if bearing < 225:
        return "S"
    return "W"


def estimate_drive_min(distance_km):
    if distance_km <= 3:
        speed = 25.0
    elif distance_km <= 12:
        speed = 40.0
    elif distance_km <= 35:
        speed = 55.0
    elif distance_km <= 100:
        speed = 72.0
    else:
        speed = 85.0
    return max(4, int(round((distance_km / speed) * 60 + 2)))


def role_for_place(place):
    text = f"{place.get('name','')} {place.get('category','')} {place.get('location','')}".lower()
    if any(k in text for k in MEAL_HINTS):
        return "meal"
    if any(k in text for k in DRINK_HINTS):
        return "drink"
    if any(k in text for k in SHOP_HINTS):
        return "shopping"
    return "activity"


def dwell_minutes(role):
    if role == "meal":
        return 95
    if role == "activity":
        return 90
    if role == "shopping":
        return 60
    if role == "drink":
        return 50
    return 75


def to_hhmm(total_minutes):
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def nearest_neighbor_order(stops, start_coord):
    remaining = stops[:]
    ordered = []
    current = start_coord
    while remaining:
        nxt = min(
            remaining,
            key=lambda s: euclidean_distance_km(current[0], current[1], s["lat"], s["lon"]),
        )
        ordered.append(nxt)
        remaining.remove(nxt)
        current = (nxt["lat"], nxt["lon"])
    return ordered


def smooth_meal_sequences(stops):
    ordered = stops[:]
    i = 1
    while i < len(ordered):
        if ordered[i - 1]["role"] == "meal" and ordered[i]["role"] == "meal":
            swap_idx = None
            for j in range(i + 1, len(ordered)):
                if ordered[j]["role"] != "meal":
                    swap_idx = j
                    break
            if swap_idx is not None:
                ordered[i], ordered[swap_idx] = ordered[swap_idx], ordered[i]
        i += 1
    return ordered


def chunk_stops_with_meal_limit(stops, max_stops, max_meals_per_plan=2):
    chunks = []
    cur = []
    cur_meals = 0
    for s in stops:
        is_meal = s["role"] == "meal"
        next_meal_count = cur_meals + (1 if is_meal else 0)
        needs_split = False
        if cur and len(cur) >= max_stops:
            needs_split = True
        if cur and is_meal:
            if next_meal_count > max_meals_per_plan:
                needs_split = True
            elif cur_meals == 1:
                first_meal_idx = next(
                    (idx for idx, x in enumerate(cur) if x["role"] == "meal"), None
                )
                has_activity_after_first_meal = any(
                    x["role"] != "meal" for x in cur[(first_meal_idx + 1) :]
                )
                if not has_activity_after_first_meal:
                    needs_split = True

        if needs_split:
            chunks.append(cur)
            cur = []
            cur_meals = 0

        cur.append(s)
        if s["role"] == "meal":
            cur_meals += 1

    if cur:
        chunks.append(cur)
    return chunks


def plan_narrative(stops):
    names = [s["name"] for s in stops]
    meal = next((s for s in stops if s["role"] == "meal"), None)
    in_between = next((s for s in stops if s["role"] != "meal"), None)
    if meal and in_between:
        return f"{in_between['name']} followed by a meal at {meal['name']}"
    if len(names) >= 2:
        return f"{names[0]} followed by {names[1]}"
    return names[0]


def build_plan(plan_id, title, zone, sector, objective, stops):
    enriched = []
    time_cursor = 9 * 60
    prev_lat, prev_lon = BASE_COORD
    total_drive = 0

    meal_indexes = [idx for idx, s in enumerate(stops) if s["role"] == "meal"]

    for i, s in enumerate(stops):
        d_km = haversine_km(prev_lat, prev_lon, s["lat"], s["lon"])
        d_min = estimate_drive_min(d_km)
        total_drive += d_min
        time_cursor += d_min

        meal_slot = None
        if s["role"] == "meal":
            if len(meal_indexes) == 1:
                meal_slot = "lunch" if time_cursor < (16 * 60) else "dinner"
            else:
                meal_slot = "lunch" if i == meal_indexes[0] else "dinner"

        enriched.append(
            {
                "order": i + 1,
                "placeIndex": s["index"],
                "placeName": s["name"],
                "category": s.get("category") or s.get("location") or "",
                "rating": s.get("rating"),
                "lat": s["lat"],
                "lon": s["lon"],
                "role": s["role"],
                "mealSlot": meal_slot,
                "plannedTime": to_hhmm(time_cursor),
                "estimatedDriveMinFromPrevious": d_min,
                "partOfPlan": "",
                "visited": False,
            }
        )

        time_cursor += dwell_minutes(s["role"])
        prev_lat, prev_lon = s["lat"], s["lon"]

    back_km = haversine_km(prev_lat, prev_lon, BASE_COORD[0], BASE_COORD[1])
    back_min = estimate_drive_min(back_km)
    total_drive += back_min

    narrative = plan_narrative(stops)
    for e in enriched:
        e["partOfPlan"] = narrative

    ratings = [e["rating"] for e in enriched if isinstance(e["rating"], (int, float))]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
    high_rated = sum(1 for r in ratings if r >= 4.5)

    meal_stops = [e for e in enriched if e["role"] == "meal"]
    activity_between = True
    if len(meal_stops) > 1:
        first = min(m["order"] for m in meal_stops)
        last = max(m["order"] for m in meal_stops)
        activity_between = any(e["role"] != "meal" and first < e["order"] < last for e in enriched)

    max_dist = max(
        haversine_km(BASE_COORD[0], BASE_COORD[1], s["lat"], s["lon"]) for s in stops
    )
    feasible = total_drive <= 320 and max_dist <= 120
    if zone == "outlier":
        feasible = False

    return {
        "planId": plan_id,
        "title": title,
        "startAddress": BASE_ADDRESS,
        "zone": zone,
        "direction": sector,
        "objective": objective,
        "isFeasibleDayTrip": feasible,
        "stops": enriched,
        "estimatedReturnDriveMin": back_min,
        "estimatedTotalDriveMin": total_drive,
        "averageStopRating": avg_rating,
        "highRatedStopCount": high_rated,
        "constraintsCheck": {
            "mealStopCount": len(meal_stops),
            "mealSlots": [m["mealSlot"] for m in meal_stops],
            "activityBetweenMeals": activity_between,
            "validMealSpacingRule": (
                len(meal_stops) <= 1
                or (
                    set(m["mealSlot"] for m in meal_stops) == {"lunch", "dinner"}
                    and activity_between
                )
            ),
        },
        "visited": False,
    }


def main():
    places_obj = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
    places = places_obj["items"]

    prepared = []
    for p in places:
        lat = as_float(p.get("lat"))
        lon = as_float(p.get("lon"))
        if lat is None or lon is None:
            continue
        rating = as_float(p.get("rating"))
        dist = haversine_km(BASE_COORD[0], BASE_COORD[1], lat, lon)
        bearing = bearing_deg(BASE_COORD[0], BASE_COORD[1], lat, lon)
        sec = sector_from_bearing(bearing)
        prepared.append(
            {
                "index": int(p["index"]),
                "name": p["name"],
                "category": p.get("category", ""),
                "location": p.get("location", ""),
                "rating": rating,
                "lat": lat,
                "lon": lon,
                "distanceKm": dist,
                "bearing": bearing,
                "sector": sec,
                "role": role_for_place(p),
            }
        )

    local = [p for p in prepared if p["distanceKm"] <= 12]
    directional = [p for p in prepared if 12 < p["distanceKm"] <= 120]
    outliers = [p for p in prepared if p["distanceKm"] > 120]

    groups = []

    dir_groups = defaultdict(list)
    for p in directional:
        dir_groups[quadrant_from_bearing(p["bearing"])].append(p)

    outlier_groups = defaultdict(list)
    for p in outliers:
        outlier_groups[quadrant_from_bearing(p["bearing"])].append(p)

    if local:
        groups.append(("local", "LOCAL", local, 5))
    for sec in ["N", "E", "S", "W"]:
        if sec in dir_groups:
            groups.append(("directional", sec, dir_groups[sec], 5))
    for sec in ["N", "E", "S", "W"]:
        if sec in outlier_groups:
            groups.append(("outlier", sec, outlier_groups[sec], 2))

    plans = []
    plan_counter = 1
    zone_counter = defaultdict(int)

    for zone, sec, raw_stops, max_stops in groups:
        ordered = nearest_neighbor_order(raw_stops, BASE_COORD)
        ordered = smooth_meal_sequences(ordered)
        chunks = chunk_stops_with_meal_limit(ordered, max_stops=max_stops, max_meals_per_plan=2)

        for chunk in chunks:
            zone_counter[(zone, sec)] += 1
            idx = zone_counter[(zone, sec)]
            sec_label = COMPASS_LABEL.get(sec, sec)

            if zone == "local":
                title = f"Gandia Core Loop {idx}"
                objective = (
                    "Keep this in-town cluster tight to minimize daily drive while clearing nearby stops."
                )
            elif zone == "directional":
                title = f"{sec_label} Direction Day Trip {idx}"
                objective = (
                    f"Cover places to the {sec_label.lower()} of Gandia in one outbound loop "
                    f"to avoid cross-direction backtracking."
                )
            else:
                title = f"{sec_label} Outlier Validation {idx}"
                objective = (
                    f"Extreme-distance points grouped by {sec_label.lower()} direction; verify coordinates "
                    f"before treating as normal day trips."
                )

            plan_id = f"day-{plan_counter:02d}"
            plans.append(build_plan(plan_id, title, zone, sec, objective, chunk))
            plan_counter += 1

    # Ensure every place is assigned exactly once in plans.
    assigned = []
    for plan in plans:
        for s in plan["stops"]:
            assigned.append(s["placeIndex"])
    assigned_set = set(assigned)
    all_indices = {p["index"] for p in prepared}
    if assigned_set != all_indices:
        missing = sorted(all_indices - assigned_set)
        extra = sorted(x for x in assigned if assigned.count(x) > 1)
        raise RuntimeError(f"Coverage mismatch. missing={missing} duplicate_entries={extra[:10]}")

    if len(assigned) != len(assigned_set):
        raise RuntimeError("Duplicate place assignment detected in plan stops.")

    for plan in plans:
        if not plan["constraintsCheck"]["validMealSpacingRule"]:
            raise RuntimeError(f"Meal spacing rule violated: {plan['planId']}")

    PLANS_PATH.write_text(json.dumps(plans, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Update place-level cross-references.
    refs = defaultdict(list)
    for plan in plans:
        for stop in plan["stops"]:
            line = f"{plan['planId']} | {plan['title']}: {stop['partOfPlan']}"
            refs[stop["placeIndex"]].append(line)

    for item in places_obj["items"]:
        idx = int(item["index"])
        item["partOfPlan"] = refs.get(idx, [])
        if not isinstance(item.get("visited"), bool):
            item["visited"] = False

    PLACES_PATH.write_text(
        json.dumps(places_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    total_drive = sum(p["estimatedTotalDriveMin"] for p in plans)
    feasible_count = sum(1 for p in plans if p["isFeasibleDayTrip"])
    print(f"wrote {PLANS_PATH} with {len(plans)} plans")
    print(f"feasible={feasible_count} non_feasible={len(plans)-feasible_count}")
    print(f"combined_estimated_drive_min={total_drive}")
    print(
        "zone_counts",
        {
            "local": sum(1 for p in plans if p["zone"] == "local"),
            "directional": sum(1 for p in plans if p["zone"] == "directional"),
            "outlier": sum(1 for p in plans if p["zone"] == "outlier"),
        },
    )


if __name__ == "__main__":
    main()
