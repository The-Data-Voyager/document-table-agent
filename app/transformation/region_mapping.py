"""Reviewed state/entity-to-region mapping for the sample power tables.

Keys match the English labels produced by ``remove_devanagari_text``. Keeping
this domain configuration separate from generic transformation code makes the
classification explicit and reviewable when a new source format is added.
"""

from types import MappingProxyType


REGION_MEMBERS = MappingProxyType(
    {
        "NR": (
            "Punjab",
            "Haryana",
            "Rajasthan",
            "Delhi",
            "UP",
            "Uttarakhand",
            "HP",
            "J&K",
            "Chandigarh",
            "Railways_NR ISTS",
        ),
        "WR": (
            "Chhattisgarh",
            "Gujarat",
            "MP",
            "Maharashtra",
            "Goa",
            "DNHDDPDCL",
            "AMNSIL",
            "BALCO",
            "RIL Jamnagar",
        ),
        "SR": (
            "Andhra Pradesh",
            "Telangana",
            "Karnataka",
            "Kerala",
            "Tamil Nadu",
            "Pondy",
        ),
        "ER": (
            "Bihar",
            "DVC",
            "Jharkhand",
            "Odisha",
            "West Bengal",
            "Sikkim",
            "Railways_ER ISTS",
        ),
        "NER": (
            "Arunachal Pradesh",
            "Assam",
            "Manipur",
            "Meghalaya",
            "Mizoram",
            "Nagaland",
            "Tripura",
        ),
    }
)

_member_count = sum(len(members) for members in REGION_MEMBERS.values())
_state_to_region = {
    member: region
    for region, members in REGION_MEMBERS.items()
    for member in members
}
if len(_state_to_region) != _member_count:
    raise RuntimeError("A state/entity appears in more than one region.")

STATE_TO_REGION = MappingProxyType(_state_to_region)
