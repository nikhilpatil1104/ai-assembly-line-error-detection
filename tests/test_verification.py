from assembly.models import Detection
from assembly.settings import get_layout
from assembly.verification import verify, batch_quality

W, H = 900, 600


def det(label, x, y, conf=.9):
    return Detection(label, conf, (x - 30, y - 30, x + 30, y + 30))


def assembled():
    return [det("red_switch", 225, 315),
            det("blue_connector", 468, 315),
            det("yellow_latch", 710, 315)]


def test_pass_all_three_parts_correctly_located():
    result = verify(assembled(), (W, H), get_layout())
    assert result["status"] == "Pass"
    assert not result["reason_codes"]


def test_missing_component():
    result = verify(assembled()[:2], (W, H), get_layout())
    assert result["status"] == "Needs Review"
    assert result["missing"] == {"yellow_latch": 1}
    assert "missing_component" in result["reason_codes"]


def test_wrong_zone_component_detected_but_not_missing():
    items = [det("red_switch", 350, 315)] + assembled()[1:]
    result = verify(items, (W, H), get_layout())
    assert result["status"] == "Needs Review"
    assert result["missing"] == {}
    assert result["misplaced"][0]["component"] == "red_switch"
    assert "misplacement" in result["reason_codes"]


def test_weak_component_does_not_count_as_valid():
    items = [det("red_switch", 225, 315, .31)] + assembled()[1:]
    result = verify(items, (W, H), get_layout())
    assert result["uncertain"] is True
    assert result["missing"] == {"red_switch": 1}
    assert "uncertain_detection" in result["reason_codes"]


def test_no_detections_flags_all_required():
    result = verify([], (W, H), get_layout())
    assert result["status"] == "Needs Review"
    assert result["missing"] == {x: 1 for x in get_layout()["components"]}


def test_duplicate_component_flags_extra():
    result = verify(assembled() + [det("blue_connector", 478, 320)], (W, H), get_layout())
    assert result["extra"] == {"blue_connector": 1}
    assert "extra_component" in result["reason_codes"]


def test_unknown_detection_requires_review():
    result = verify(assembled() + [det("unconfigured_part", 400, 340)], (W, H), get_layout())
    assert "unknown_component" in result["reason_codes"]


def test_exactly_ten_percent_does_not_warn():
    statuses = ["Pass"] * 9 + ["Needs Review"]
    result = batch_quality(statuses)
    assert result["flagged_rate"] == .1
    assert result["batch_quality_warning"] is False


def test_more_than_ten_percent_warns():
    statuses = ["Pass"] * 8 + ["Needs Review"] * 2
    result = batch_quality(statuses)
    assert result["flagged_rate"] == .2
    assert result["batch_quality_warning"] is True


def test_empty_batch_safe():
    result = batch_quality([])
    assert result["inspected_units"] == 0
    assert result["batch_quality_warning"] is False
