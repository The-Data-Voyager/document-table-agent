from app.transformation.region_mapping import REGION_MEMBERS, STATE_TO_REGION


def test_region_mapping_contains_each_member_exactly_once():
    members = [
        member
        for region_members in REGION_MEMBERS.values()
        for member in region_members
    ]

    assert len(members) == 39
    assert len(set(members)) == 39
    assert len(STATE_TO_REGION) == 39


def test_region_mapping_has_expected_group_sizes():
    assert {region: len(members) for region, members in REGION_MEMBERS.items()} == {
        "NR": 10,
        "WR": 9,
        "SR": 6,
        "ER": 7,
        "NER": 7,
    }
