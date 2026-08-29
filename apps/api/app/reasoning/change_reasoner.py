from __future__ import annotations


def explain_change(stats: dict[str, float], target: str | None = None, *, learned: bool = False) -> str:
    changed = stats["changed_area_percent"]
    built = stats["built_up_change_pp"]
    vegetation = stats["vegetation_change_pp"]
    water = stats["water_change_pp"]
    if target == "built_up":
        direction = "increased" if built > 0.5 else "decreased" if built < -0.5 else "remained broadly stable"
        change_type = "ChangeFormer change evidence" if learned else "material spectral change"
        return f"Built-up evidence {direction} by {abs(built):.1f} percentage points. {changed:.1f}% of pixels show {change_type}."
    if target == "vegetation":
        direction = "increased" if vegetation > 0.5 else "decreased" if vegetation < -0.5 else "remained broadly stable"
        change_type = "ChangeFormer change evidence" if learned else "material spectral change"
        return f"Vegetation evidence {direction} by {abs(vegetation):.1f} percentage points. {changed:.1f}% of pixels show {change_type}."
    if target == "water":
        direction = "increased" if water > 0.5 else "decreased" if water < -0.5 else "remained broadly stable"
        change_type = "ChangeFormer change evidence" if learned else "material spectral change"
        return f"Water evidence {direction} by {abs(water):.1f} percentage points. {changed:.1f}% of pixels show {change_type}."
    source = "learned ChangeFormer and land-cover evidence" if learned else "the deterministic baseline"
    return (
        f"Material change is visible across {changed:.1f}% of the scene. "
        f"{source.capitalize()} estimates built-up {built:+.1f} pp, vegetation {vegetation:+.1f} pp, and water {water:+.1f} pp."
    )
