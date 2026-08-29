from __future__ import annotations


def explain_change(
    stats: dict[str, float | str],
    target: str | None = None,
    *,
    learned: bool = False,
    alignment_method: str = "not_provided",
) -> str:
    changed = float(stats["changed_area_percent"])
    built = float(stats["built_up_change_pp"])
    vegetation = float(stats["vegetation_change_pp"])
    water = float(stats["water_change_pp"])
    approximate = alignment_method in {"pixel_space_resize", "native_pixel_grid"}
    area_phrase = f"Approximately {changed:.1f}%" if approximate else f"{changed:.1f}%"
    if alignment_method == "pixel_space_resize":
        qualification = " Area estimates are approximate because the unreferenced images required pixel-space resizing."
    elif alignment_method == "native_pixel_grid":
        qualification = " Area estimates are pixel-space approximations because the imagery has no shared georeferencing."
    else:
        qualification = ""
    method = str(stats.get("change_method", "material_change"))
    location = str(stats.get("largest_change_location", "strongest connected image region"))
    location_clause = f"near the {location}" if location.startswith("projected coordinate") else f"in the {location}"
    change_type = (
        "hybrid environmental and semantic change evidence"
        if method == "environmental_semantic_hybrid"
        else "structural and semantic change evidence"
    )
    if target == "built_up":
        direction = "increased" if built > 0.5 else "decreased" if built < -0.5 else "remained broadly stable"
        return f"Built-up evidence {direction} by {abs(built):.1f} percentage points. {area_phrase} of valid pixels show {change_type}; the largest region is {location_clause}.{qualification}"
    if target == "vegetation":
        direction = "increased" if vegetation > 0.5 else "decreased" if vegetation < -0.5 else "remained broadly stable"
        return f"Vegetation evidence {direction} by {abs(vegetation):.1f} percentage points. {area_phrase} of valid pixels show {change_type}; the largest region is {location_clause}.{qualification}"
    if target == "water":
        direction = "increased" if water > 0.5 else "decreased" if water < -0.5 else "remained broadly stable"
        return f"Water evidence {direction} by {abs(water):.1f} percentage points. {area_phrase} of valid pixels show {change_type}; the largest region is {location_clause}.{qualification}"

    changes = {
        "water": water,
        "vegetation": vegetation,
        "built-up": built,
        "bare land": float(stats.get("bare_land_change_pp", 0.0)),
        "agriculture": float(stats.get("agriculture_change_pp", 0.0)),
    }
    strongest_label, strongest_value = max(changes.items(), key=lambda item: abs(item[1]))
    strongest_direction = "increased" if strongest_value > 0 else "decreased"
    source = "ChangeFormer V6 and SatQuery land-cover evidence" if learned else "Environmental appearance and semantic baseline evidence"
    return (
        f"{area_phrase} of the valid scene shows material change, with the largest region {location_clause}. {source} identifies the strongest land-cover signal as "
        f"{strongest_label} {strongest_direction} by {abs(strongest_value):.1f} percentage points."
        f"{qualification}"
    )
