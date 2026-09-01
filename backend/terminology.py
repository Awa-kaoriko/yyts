"""Terminology correction and translation checks for reviewed cultural terms."""

from __future__ import annotations

import json
import re
from pathlib import Path


class Terminology:
    def __init__(self, glossary_path: Path) -> None:
        data = json.loads(glossary_path.read_text(encoding="utf-8"))
        self.terms = [
            item
            for item in data
            if item.get("status") == "reviewed" and item.get("zh") and item.get("en")
        ]
        self._alias_map = {
            alias: term
            for term in self.terms
            for alias in [term["zh"], *term.get("aliases", [])]
            if alias
        }
        self._aliases = sorted(self._alias_map, key=len, reverse=True)

    def correct(self, source_text: str) -> tuple[str, list[dict[str, str]]]:
        corrected = source_text
        for alias in self._aliases:
            term = self._alias_map[alias]
            corrected = corrected.replace(alias, term.get("asr_text", term["zh"]))

        matches: list[dict[str, str]] = []
        occupied: list[tuple[int, int]] = []
        for alias in self._aliases:
            term = self._alias_map[alias]
            start = corrected.find(alias)
            if start < 0:
                continue
            end = start + len(alias)
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
            matches.append(
                {
                    "zh": term["zh"],
                    "en": term["en"],
                    "tts_reading": term.get("tts_reading", ""),
                    "forbidden": ", ".join(term.get("forbidden", [])),
                    "source": term.get("source", ""),
                }
            )
        return corrected, matches

    @staticmethod
    def tts_text(translated_text: str, matches: list[dict[str, str]]) -> str:
        spoken = translated_text
        for term in matches:
            if term["tts_reading"]:
                spoken = re.sub(
                    re.escape(term["en"]), term["tts_reading"], spoken, flags=re.IGNORECASE
                )
        return spoken

    @staticmethod
    def validate(
        translated_text: str, subtitle_text: str, tts_text: str, matches: list[dict[str, str]]
    ) -> list[str]:
        errors: list[str] = []
        visible_text = f"{translated_text}\n{subtitle_text}".casefold()
        all_text = f"{visible_text}\n{tts_text}".casefold()
        for term in matches:
            if term["en"].casefold() not in visible_text:
                errors.append(f"{term['zh']} must use '{term['en']}'.")
            for forbidden in filter(None, (value.strip() for value in term["forbidden"].split(","))):
                if forbidden.casefold() in all_text:
                    errors.append(f"'{forbidden}' is forbidden for {term['zh']}.")
        return errors
