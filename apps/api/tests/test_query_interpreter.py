from app.agent.query_interpreter import QueryInterpreter


def test_routes_grounding_without_user_model_selection():
    result = QueryInterpreter().classify("Highlight the largest water body", "single")
    assert result.intent == "REGION_GROUNDING"
    assert result.entities["target_class"] == "water"
    assert "grounding" in result.required_capabilities


def test_routes_built_up_change():
    result = QueryInterpreter().classify("Has the built-up area increased?", "bi_temporal")
    assert result.intent == "BUILT_UP_CHANGE"
    assert result.entities["target_class"] == "built_up"


def test_routes_optical_sar_water():
    result = QueryInterpreter().classify("Use both sensors to identify water", "cross_modal")
    assert result.intent == "OPTICAL_SAR_WATER"
