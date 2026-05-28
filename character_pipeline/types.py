from typing import TypedDict


class CharacterRecord(TypedDict):
    aliases: list[str]
    positions: list[int]


class CharacterAliasesOnly(TypedDict):
    aliases: list[str]
