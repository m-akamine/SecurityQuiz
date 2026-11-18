"""Question validation and conservative, non-biometric face-position tracking."""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class QuestionDataError(ValueError):
    """Raised when the question file does not match the published format."""


class TrackingError(ValueError):
    """Raised when face positions cannot be associated safely."""


class ExtraFacesError(TrackingError):
    """Raised when more faces are detected than the registered participants."""


@dataclass(frozen=True)
class Detection:
    center_x: float
    center_y: float
    box: tuple[float, float, float, float]


@dataclass
class TrackedFace:
    track_id: int
    detection: Detection
    visible: bool = True
    last_seen_at: float = 0.0
    missing_since: float | None = None
    lost: bool = False


def load_question_sets(path: Path) -> dict[int, list[dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig") as question_file:
            data = json.load(question_file)
    except FileNotFoundError as error:
        raise QuestionDataError(f"問題ファイルが見つかりません: {path}") from error
    except json.JSONDecodeError as error:
        raise QuestionDataError(
            f"問題JSONの形式が正しくありません（{error.lineno}行目）"
        ) from error
    except UnicodeError as error:
        raise QuestionDataError("問題ファイルはUTF-8形式で保存してください。") from error
    except OSError as error:
        raise QuestionDataError(f"問題ファイルを読み込めません: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("sets"), list):
        raise QuestionDataError("JSONの最上位に sets の配列が必要です。")

    sets = data["sets"]
    if not sets:
        raise QuestionDataError("問題セットは1種類以上必要です。")

    result: dict[int, list[dict[str, Any]]] = {}
    for expected_set_id, question_set in enumerate(sets, start=1):
        if not isinstance(question_set, dict):
            raise QuestionDataError(f"{expected_set_id}番目のセットが不正です。")
        if (
            type(question_set.get("set_id")) is not int
            or question_set["set_id"] != expected_set_id
        ):
            raise QuestionDataError("セット番号は1から順番に指定してください。")

        questions = question_set.get("questions")
        if not isinstance(questions, list) or len(questions) != 5:
            raise QuestionDataError(f"セット{expected_set_id}は5問必要です。")

        levels: set[int] = set()
        for question in questions:
            if not isinstance(question, dict):
                raise QuestionDataError(f"セット{expected_set_id}に不正な問題があります。")
            level = question.get("level")
            if type(level) is not int or level not in range(1, 6) or level in levels:
                raise QuestionDataError(
                    f"セット{expected_set_id}の難易度は★1〜★5を1問ずつ指定してください。"
                )
            levels.add(level)
            for key, label in (
                ("question", "問題文"),
                ("explanation", "解説"),
            ):
                if not isinstance(question.get(key), str) or not question[key].strip():
                    raise QuestionDataError(
                        f"セット{expected_set_id}・★{level}の{label}が空です。"
                    )
            if question.get("answer") not in ("〇", "×"):
                raise QuestionDataError(
                    f"セット{expected_set_id}・★{level}の正解は〇か×にしてください。"
                )

        if levels != {1, 2, 3, 4, 5}:
            raise QuestionDataError(f"セット{expected_set_id}の難易度が不足しています。")
        result[expected_set_id] = sorted(questions, key=lambda item: item["level"])

    return result


class FaceTracker:
    """Match nearby face positions only; never identifies a person by appearance."""

    MISSED_SECONDS_GRACE = 3.0

    def __init__(self, initial_detections: list[Detection]) -> None:
        if not 1 <= len(initial_detections) <= 5:
            raise TrackingError("参加者の顔を1〜5人の範囲で確認できません。")
        initial = sorted(initial_detections, key=lambda face: face.center_x)
        now = time.monotonic()
        self.tracks = {
            index: TrackedFace(index, detection, last_seen_at=now)
            for index, detection in enumerate(initial, start=1)
        }

    def update(self, detections: list[Detection]) -> None:
        now = time.monotonic()
        if len(detections) > len(self.tracks):
            raise ExtraFacesError("新しい顔または追跡の入れ替わりを検出しました。")

        track_ids = list(self.tracks)
        assignments: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for chosen_tracks in itertools.combinations(track_ids, len(detections)):
            for detection_order in itertools.permutations(range(len(detections))):
                distances = tuple(
                    math.hypot(
                        self.tracks[track_id].detection.center_x
                        - detections[detection_index].center_x,
                        self.tracks[track_id].detection.center_y
                        - detections[detection_index].center_y,
                    )
                    for track_id, detection_index in zip(
                        chosen_tracks, detection_order
                    )
                )
                assignments.append(
                    (sum(distances), chosen_tracks, detection_order)
                )

        if not assignments:
            for track in self.tracks.values():
                track.visible = False
                if track.missing_since is None:
                    track.missing_since = now
            return

        assignments.sort(key=lambda item: item[0])
        best = assignments[0]
        assigned_detections: dict[int, Detection] = {}
        for track_id, detection_index in zip(best[1], best[2]):
            assigned_detections[track_id] = detections[detection_index]

        for track_id, track in self.tracks.items():
            detection = assigned_detections.get(track_id)
            if detection is None:
                track.visible = False
                if track.missing_since is None:
                    track.missing_since = now
            else:
                track.detection = detection
                track.visible = True
                track.last_seen_at = now
                track.missing_since = None

    def visible_ids(self) -> set[int]:
        now = time.monotonic()
        return {
            track_id
            for track_id, track in self.tracks.items()
            if (
                not track.lost
                and now - track.last_seen_at <= self.MISSED_SECONDS_GRACE
            )
        }

    def detected_ids(self) -> set[int]:
        return {
            track_id for track_id in self.visible_ids()
            if self.tracks[track_id].visible
        }

    def answer(self, correct_answer: str) -> tuple[set[int], set[int]]:
        visible = self.detected_ids()
        if not visible:
            return set(), set(self.tracks)
        correct_ids = {
            track_id
            for track_id in visible
            if (
                self.tracks[track_id].detection.center_x < 0.5
                and correct_answer == "〇"
            )
            or (
                self.tracks[track_id].detection.center_x >= 0.5
                and correct_answer == "×"
            )
        }
        return correct_ids, set(self.tracks) - correct_ids

    def retain(self, track_ids: set[int]) -> None:
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track_id in track_ids
        }
