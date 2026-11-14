"""Windows desktop security quiz. Camera frames remain in memory only."""

from __future__ import annotations

import argparse
import base64
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tkinter import font as tkfont, messagebox, ttk
from typing import Any

try:
    import cv2
    from PIL import Image, ImageDraw, ImageTk
except ImportError as error:
    cv2 = None
    Image = ImageDraw = ImageTk = None
    _DEPENDENCY_ERROR = error
else:
    _DEPENDENCY_ERROR = None

from quiz_core import (
    Detection,
    ExtraFacesError,
    FaceTracker,
    QuestionDataError,
    TrackingError,
    load_question_sets,
)


APP_TITLE = "セキュリティ〇×クイズ"
QUESTION_FILE = Path(__file__).with_name("261113セキュリティクイズ問題.json")
BACKGROUND_FILE = Path(__file__).with_name("BackgroundImage_Clean.png")
PANEL = "#0B1D26"
SUBPANEL = "#102A36"
CYAN = "#00E5FF"
AMBER = "#FBBF24"
WHITE = "#F0F9FF"
PALE_CYAN = "#BAE6FD"
GREEN = "#059669"
RED = "#E11D48"
MAGENTA = "#FF00FF"
START_COUNTDOWN_SECONDS = 10
RESULT_SCREEN_SECONDS = 5
WIN_SCREEN_SECONDS = 10
QUESTION_INTRO_SECONDS = 2
FACE_RECOVERY_SECONDS = 10
COUNTDOWN_FACE_LOSS_GRACE_SECONDS = 2.0
ANSWER_FACE_RETRY_SECONDS = 2.0
EXTRA_FACE_RETRY_SECONDS = 2.0
FONT_CANDIDATES = (
    "Noto Sans JP", "BIZ UDPゴシック", "BIZ UDPGothic", "Yu Gothic UI", "Meiryo UI"
)


def choose_japanese_font(available: tuple[str, ...]) -> str:
    return next((name for name in FONT_CANDIDATES if name in available), "Yu Gothic UI")


class CameraEnumerationError(RuntimeError):
    """Raised when Windows camera names cannot be enumerated."""


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str

    @property
    def label(self) -> str:
        return f"{self.name}（カメラ {self.index}）"


def devices_from_names(names: list[str]) -> list[CameraDevice]:
    return [
        CameraDevice(index=index, name=name.strip())
        for index, name in enumerate(names)
        if name.strip()
    ]


def merge_face_boxes(
    *groups: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for group in groups:
        for box in group:
            x, y, width, height = box
            area = width * height
            duplicate = False
            for existing in merged:
                left = max(x, existing[0])
                top = max(y, existing[1])
                right = min(x + width, existing[0] + existing[2])
                bottom = min(y + height, existing[1] + existing[3])
                intersection = max(0, right - left) * max(0, bottom - top)
                union = area + existing[2] * existing[3] - intersection
                if union and intersection / union >= 0.35:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(box)
    return sorted(merged, key=lambda box: box[0])


def detect_faces(
    gray: Any,
    frontal_detector: Any,
    profile_detector: Any,
    tracked_boxes: tuple[tuple[float, float, float, float], ...] | None,
) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape[:2]
    regions = []
    if tracked_boxes is None:
        regions.append((0, 0, width, height))
    else:
        for x, y, box_width, box_height in tracked_boxes:
            center_x = x + box_width / 2
            center_y = y + box_height / 2
            radius_x = max(0.18, 2.5 * box_width)
            radius_y = max(0.12, 2.0 * box_height)
            left = max(0, math.floor((center_x - radius_x) * width))
            top = max(0, math.floor((center_y - radius_y) * height))
            right = min(width, math.ceil((center_x + radius_x) * width))
            bottom = min(height, math.ceil((center_y + radius_y) * height))
            if right > left and bottom > top:
                regions.append((left, top, right, bottom))

    groups: list[list[tuple[int, int, int, int]]] = []
    for left, top, right, bottom in regions:
        region = gray[top:bottom, left:right]
        frontal = frontal_detector.detectMultiScale(
            region, scaleFactor=1.12, minNeighbors=6, minSize=(40, 40)
        )
        profile = profile_detector.detectMultiScale(
            region, scaleFactor=1.12, minNeighbors=8, minSize=(40, 40)
        )
        flipped_profile = profile_detector.detectMultiScale(
            cv2.flip(region, 1),
            scaleFactor=1.12,
            minNeighbors=8,
            minSize=(40, 40),
        )
        groups.extend(
            [
                [
                    (left + int(x), top + int(y), int(w), int(h))
                    for x, y, w, h in frontal
                ],
                [
                    (left + int(x), top + int(y), int(w), int(h))
                    for x, y, w, h in profile
                ],
                [
                    (right - int(x) - int(w), top + int(y), int(w), int(h))
                    for x, y, w, h in flipped_profile
                ],
            ]
        )
    return merge_face_boxes(*groups)


def enumerate_camera_devices() -> list[CameraDevice]:
    try:
        filter_graph = import_module("pygrabber.dshow_graph").FilterGraph
    except ImportError as error:
        raise CameraEnumerationError(
            "カメラ一覧の取得に必要なライブラリがありません。"
            "「python -m pip install -r requirements.txt」を実行してください。"
        ) from error

    try:
        names = filter_graph().get_input_devices()
    except Exception as error:
        raise CameraEnumerationError(
            f"Windowsのカメラ一覧を取得できません: {error}"
        ) from error

    return devices_from_names(names)


SPEECH_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($voiceName)
$synth.Speak($text)
$synth.Dispose()
"""
VOICE_CHECK_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() |
    Where-Object { $_.VoiceInfo.Culture.Name -eq 'ja-JP' } |
    Select-Object -First 1
if ($null -eq $voice) { exit 2 }
[Console]::Write($voice.VoiceInfo.Name)
$synth.Dispose()
"""


def powershell_command(script: str) -> list[str]:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded,
    ]


class LocalSpeech:
    def __init__(self) -> None:
        if shutil.which("powershell.exe") is None:
            raise RuntimeError("Windows PowerShellが見つかりません。")
        result = subprocess.run(
            powershell_command(VOICE_CHECK_SCRIPT),
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                "日本語のローカル音声が利用できません。"
                "Windowsの音声設定で日本語音声を追加してください。"
            )
        self.voice_name = result.stdout.strip()
        self.process: subprocess.Popen[str] | None = None

    def speak(self, text: str) -> None:
        self.stop()
        encoded_text = base64.b64encode(text.encode("utf-16le")).decode("ascii")
        encoded_voice = base64.b64encode(
            self.voice_name.encode("utf-16le")
        ).decode("ascii")
        script = (
            "$text = [System.Text.Encoding]::Unicode.GetString("
            "[System.Convert]::FromBase64String('"
            + encoded_text
            + "')); $voiceName = [System.Text.Encoding]::Unicode.GetString("
            "[System.Convert]::FromBase64String('"
            + encoded_voice
            + "')); "
            + SPEECH_SCRIPT
        )
        self.process = subprocess.Popen(
            powershell_command(script),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def finished(self) -> bool:
        return self.process is None or self.process.poll() is not None

    def check(self) -> None:
        process = self.process
        if process is not None and process.poll() is not None:
            output, _ = process.communicate()
            self.process = None
            if process.returncode:
                detail = (output or "").strip()
                raise RuntimeError(
                    "日本語音声の読み上げに失敗しました。"
                    + (f"\n{detail}" if detail else "")
                )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.process = None


class CameraWorker:
    """Capture and detect faces on a worker; no frame is written or transmitted."""

    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="quiz-camera", daemon=True
        )
        self.frame = None
        self.faces: list[tuple[int, int, int, int]] = []
        self.sequence = 0
        self.error: str | None = None
        self.ready = False
        self._capture = None
        self._tracked_boxes: tuple[tuple[float, float, float, float], ...] | None = None
        self._regions_version = 0

    def start(self) -> None:
        self._thread.start()

    def snapshot(
        self,
    ) -> tuple[Any, list[tuple[int, int, int, int]], int, bool, str | None]:
        with self._lock:
            return self.frame, list(self.faces), self.sequence, self.ready, self.error

    def set_tracked_boxes(
        self, boxes: tuple[tuple[float, float, float, float], ...] | None
    ) -> None:
        with self._lock:
            if self._tracked_boxes != boxes:
                scope_changed = (
                    (self._tracked_boxes is None) != (boxes is None)
                    or (
                        self._tracked_boxes is not None
                        and boxes is not None
                        and len(self._tracked_boxes) != len(boxes)
                    )
                )
                self._tracked_boxes = boxes
                self._regions_version += 1
                if scope_changed:
                    self.faces = []
                    self.sequence += 1

    def stop(self) -> bool:
        self._stop.set()
        with self._lock:
            capture = self._capture
        if capture is not None:
            capture.release()
        self._thread.join(timeout=3)
        return not self._thread.is_alive()

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cascade_temp_paths: list[Path] = []
        try:
            with self._lock:
                self._capture = capture
            if not capture.isOpened():
                self._set_error(
                    f"カメラ（番号 {self.camera_index}）を開けません。"
                    "接続、Windowsのカメラ使用許可、カメラ番号を確認してください。"
                )
                return

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cascade_dir = Path(cv2.data.haarcascades)
            cascade_names = (
                "haarcascade_frontalface_default.xml",
                "haarcascade_profileface.xml",
            )
            windows_temp = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Temp"
            detectors = []
            for cascade_name in cascade_names:
                with tempfile.NamedTemporaryFile(
                    prefix="security_quiz_",
                    suffix=".xml",
                    dir=windows_temp,
                    delete=False,
                ) as model_file:
                    cascade_temp_path = Path(model_file.name)
                cascade_temp_paths.append(cascade_temp_path)
                shutil.copyfile(cascade_dir / cascade_name, cascade_temp_path)
                detector = cv2.CascadeClassifier(str(cascade_temp_path))
                if detector.empty():
                    self._set_error(f"顔検出用データを読み込めません: {cascade_name}")
                    return
                detectors.append(detector)
            if len(detectors) != 2:
                self._set_error("顔検出用データを読み込めません。")
                return
            frontal_detector, profile_detector = detectors

            while not self._stop.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    self._set_error(
                        "カメラ映像を取得できません。カメラの接続を確認してください。"
                    )
                    return

                mirrored = cv2.flip(frame, 1)
                gray = cv2.cvtColor(mirrored, cv2.COLOR_BGR2GRAY)
                with self._lock:
                    tracked_boxes = self._tracked_boxes
                    regions_version = self._regions_version
                detections = detect_faces(
                    gray, frontal_detector, profile_detector, tracked_boxes
                )
                with self._lock:
                    if regions_version != self._regions_version:
                        continue
                    self.frame = mirrored
                    self.faces = detections
                    self.sequence += 1
                    self.ready = True
                time.sleep(0.01)
        except Exception as error:
            self._set_error(f"カメラ処理に失敗しました: {error}")
        finally:
            capture.release()
            for cascade_temp_path in cascade_temp_paths:
                try:
                    cascade_temp_path.unlink(missing_ok=True)
                except OSError as error:
                    self._set_error(
                        f"顔検出用の一時ファイルを削除できません: {error}"
                    )
            with self._lock:
                self._capture = None

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.ready = False


class SecurityQuizApp:
    def __init__(self, root: tk.Tk, camera_index: int) -> None:
        self.root = root
        self.font_family = choose_japanese_font(tkfont.families(root))
        self.camera_index = camera_index
        self.root.title(APP_TITLE)
        self.root.geometry("1280x760")
        self.root.minsize(960, 600)
        self.root.state("zoomed")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=PANEL)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<space>", self._space_pressed)
        self.root.bind("<c>", self._camera_test_pressed)
        self.root.bind("<Control-Shift-q>", self._staff_cancel)
        self.root.bind("<F12>", self._staff_cancel)
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._escape_pressed)

        self.sets: dict[int, list[dict]] = {}
        self.setup_errors: list[str] = []
        try:
            self.sets = load_question_sets(QUESTION_FILE)
        except QuestionDataError as error:
            self.setup_errors.append(str(error))

        self.background_image = None
        self.background_photo = None
        self.background_size: tuple[int, int] | None = None
        if Image is not None:
            try:
                with Image.open(BACKGROUND_FILE) as image:
                    self.background_image = image.convert("RGB")
            except OSError as error:
                self.setup_errors.append(f"背景画像を読み込めません: {error}")

        self.speech: LocalSpeech | None = None
        try:
            self.speech = LocalSpeech()
        except (RuntimeError, subprocess.SubprocessError, OSError) as error:
            self.setup_errors.append(str(error))

        self.worker: CameraWorker | None = None
        self.camera_error: str | None = None
        if _DEPENDENCY_ERROR is not None:
            self.camera_error = (
                "カメラ表示に必要なPythonライブラリがありません。"
                " PowerShellで「py -3 -m pip install -r requirements.txt」を実行してください。"
            )
        else:
            try:
                self.camera_devices = enumerate_camera_devices()
            except CameraEnumerationError as error:
                self.camera_devices = []
                self.camera_error = str(error)
            if not self.camera_devices and self.camera_error is None:
                self.camera_error = (
                    "Windowsで利用可能なカメラが見つかりません。"
                    "接続を確認して「カメラ一覧を更新」を押してください。"
                )
            if self.camera_devices:
                selected = next(
                    (
                        device
                        for device in self.camera_devices
                        if device.index == camera_index
                    ),
                    self.camera_devices[0],
                )
                self.camera_index = selected.index
                self.worker = CameraWorker(selected.index)
                self.worker.start()
        if not hasattr(self, "camera_devices"):
            self.camera_devices = []

        self.canvas = tk.Canvas(
            root, bg=PANEL, highlightthickness=0, borderwidth=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="カメラを確認しています…")
        self.status_label = tk.Label(
            root,
            textvariable=self.status_var,
            font=(self.font_family, 13),
            fg=WHITE,
            bg=PANEL,
            anchor="w",
            padx=24,
            pady=8,
        )
        self.status_label.pack(fill="x")
        self.controls = ttk.Frame(root, padding=(12, 4))
        ttk.Label(self.controls, text="カメラ:").pack(side="left", padx=(8, 2))
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            self.controls,
            textvariable=self.camera_var,
            values=self._camera_labels(),
            state="readonly",
            width=32,
        )
        self.camera_combo.pack(side="left", padx=4)
        self.camera_combo.bind(
            "<<ComboboxSelected>>", self._camera_selection_changed
        )
        self.refresh_camera_button = ttk.Button(
            self.controls,
            text="カメラ一覧を更新",
            command=self.refresh_cameras,
        )
        self.refresh_camera_button.pack(side="left", padx=4)
        selected_device = self._selected_camera_device()
        if selected_device is not None:
            self.camera_var.set(selected_device.label)
        self.start_button = ttk.Button(
            self.controls, text="ゲーム開始  [Space]", command=self.start_game
        )
        self.start_button.pack(side="left", padx=8)
        self.camera_test_button = ttk.Button(
            self.controls,
            text="カメラ確認  [C]",
            command=self.toggle_camera_test,
        )
        self.camera_test_button.pack(side="left", padx=8)
        self.cancel_button = ttk.Button(
            self.controls, text="ゲーム中止  [Ctrl+Shift+Q]", command=self._staff_cancel
        )
        self.cancel_button.pack(side="left", padx=8)
        self.controls.pack(fill="x")

        self.mode = "home"
        self.home_message = ""
        self.end_message = ""
        self.next_set_id = 1
        self.current_set_id = 1
        self.question_index = 0
        self.tracker: FaceTracker | None = None
        self.phase_started = time.monotonic()
        self.last_sequence = -1
        self.latest_frame = None
        self.latest_faces: list[tuple[int, int, int, int]] = []
        self.latest_detections: list[Detection] = []
        self.correct_ids: set[int] = set()
        self.wrong_ids: set[int] = set()
        self.correct_missing_since: float | None = None
        self.extra_face_since: float | None = None
        self.face_loss_since: float | None = None

        self.canvas.bind("<Configure>", lambda _event: self._render())
        self._update()

    def _toggle_fullscreen(self, _event: tk.Event) -> str:
        fullscreen = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not fullscreen)
        return "break"

    def _space_pressed(self, _event: tk.Event) -> str | None:
        if self.mode == "home":
            self.start_game()
            return "break"
        if self.mode == "waiting_for_players":
            self._cancel_player_wait()
            return "break"
        return None

    def _camera_test_pressed(self, _event: tk.Event) -> str | None:
        if self.mode == "home":
            self.toggle_camera_test()
            return "break"
        return None

    def _camera_labels(self) -> list[str]:
        return [device.label for device in self.camera_devices]

    def _selected_camera_device(self) -> CameraDevice | None:
        return next(
            (
                device
                for device in self.camera_devices
                if device.index == self.camera_index
            ),
            None,
        )

    def _camera_selection_changed(self, _event: tk.Event) -> None:
        if self.mode != "home":
            return
        selected = next(
            (
                device
                for device in self.camera_devices
                if device.label == self.camera_var.get()
            ),
            None,
        )
        if selected is not None:
            self._switch_camera(selected)

    def refresh_cameras(self) -> None:
        if self.mode != "home":
            return
        try:
            devices = enumerate_camera_devices()
        except CameraEnumerationError as error:
            self._return_home(f"カメラ一覧: {error}")
            return

        if not devices:
            if self.worker is not None:
                if not self.worker.stop():
                    self._return_home(
                        "現在のカメラを解放できませんでした。"
                        "ほかのアプリで使用中でないか確認してください。"
                    )
                    return
                self.worker = None
            self.camera_devices = []
            self.camera_combo.configure(values=[])
            self.camera_var.set("")
            self.latest_frame = None
            self.latest_faces = []
            self.latest_detections = []
            self.camera_error = (
                "Windowsで利用可能なカメラが見つかりません。"
                "接続とカメラ使用許可を確認してください。"
            )
            self._render()
            return

        self.camera_devices = devices
        self.camera_combo.configure(values=self._camera_labels())
        selected = next(
            (device for device in devices if device.index == self.camera_index),
            devices[0],
        )
        self.camera_var.set(selected.label)
        self._switch_camera(selected)

    def _switch_camera(self, device: CameraDevice) -> None:
        if self.mode != "home":
            self.camera_var.set(
                (self._selected_camera_device() or device).label
            )
            return
        current_device = self._selected_camera_device()
        if (
            current_device == device
            and self.worker is not None
            and self.camera_error is None
        ):
            return
        if self.worker is not None and not self.worker.stop():
            self._return_home(
                "現在のカメラを解放できませんでした。"
                "ほかのアプリで使用中でないか確認してください。"
            )
            if current_device is not None:
                self.camera_var.set(current_device.label)
            return

        self.worker = None
        self.latest_frame = None
        self.latest_faces = []
        self.latest_detections = []
        self.last_sequence = -1
        self.camera_index = device.index
        self.camera_var.set(device.label)
        self.camera_error = None
        self.worker = CameraWorker(device.index)
        self.worker.start()
        self._render()

    def _escape_pressed(self, _event: tk.Event) -> str | None:
        if self.mode == "camera_test":
            self.mode = "home"
            self._render()
            return "break"
        return None

    def toggle_camera_test(self) -> None:
        if self.mode == "camera_test":
            self.mode = "home"
            self._render()
            return
        if self.mode != "home":
            return
        if self.worker is None or self.camera_error:
            self._return_home(self.camera_error or "カメラを利用できません。")
            return
        frame, _faces, _sequence, ready, error = self.worker.snapshot()
        if not ready or frame is None:
            self._return_home(
                error or "カメラの準備ができていません。少し待ってから再試行してください。",
            )
            return
        self.mode = "camera_test"
        self._render()

    def _staff_cancel(self, _event: tk.Event | None = None) -> None:
        if self.mode == "home":
            return
        if messagebox.askyesno(
            "ゲーム中止",
            "ゲームを中止してホーム画面へ戻りますか？\n次のプレイは次のセットから始まります。",
            parent=self.root,
        ):
            self._return_home("スタッフがゲームを中止しました。")

    def start_game(self) -> None:
        if self.mode != "home":
            return
        if self.setup_errors:
            self._return_home("\n".join(self.setup_errors))
            return
        if self.worker is None or self.camera_error:
            self._return_home(self.camera_error or "カメラを利用できません。")
            return
        frame, _faces, _sequence, ready, error = self.worker.snapshot()
        if not ready or frame is None:
            self._return_home(
                error or "カメラの準備ができていません。少し待ってから再試行してください。",
            )
            return
        self.mode = "waiting_for_players"
        self.home_message = "参加者の顔を確認しています。"
        self.tracker = None
        if 1 <= len(self.latest_detections) <= 5:
            self._start_countdown()
        self._render()

    def _update(self) -> None:
        if self.worker is not None:
            frame, faces, sequence, ready, error = self.worker.snapshot()
            if error and self.camera_error is None:
                self.camera_error = error
                if self.mode != "home":
                    self._return_home(error)
            if ready and frame is not None and self.camera_error is None:
                self.latest_frame = frame
                self.latest_faces = faces
                if sequence != self.last_sequence:
                    self.last_sequence = sequence
                    self.latest_detections = self._make_detections(faces, frame)
                    self._process_new_frame()

        if self.mode in (
            "question",
            "question_intro",
            "answer",
            "reveal",
            "explanation",
            "exit",
            "countdown",
            "notice",
            "win",
            "lose",
        ):
            self._advance_phase()

        self.status_var.set(self._status_text())
        self._render()
        self.root.after(35, self._update)

    def _make_detections(
        self,
        faces: list[tuple[int, int, int, int]],
        frame: Any,
    ) -> list[Detection]:
        height, width = frame.shape[:2]
        return [
            Detection(
                center_x=(x + box_width / 2) / width,
                center_y=(y + box_height / 2) / height,
                box=(x / width, y / height, box_width / width, box_height / height),
            )
            for x, y, box_width, box_height in faces
        ]

    def _process_new_frame(self) -> None:
        if self.mode == "waiting_for_players":
            if 1 <= len(self.latest_detections) <= 5:
                self._start_countdown()
            elif len(self.latest_detections) > 5:
                self.home_message = "参加できる人数は1〜5人です。"
            return

        if self.mode == "home":
            if self.setup_errors or self.camera_error:
                return
            if 1 <= len(self.latest_detections) <= 5:
                self._start_countdown()
            elif len(self.latest_detections) > 5:
                self.home_message = "参加できる人数は1〜5人です。"
            return

        if self.mode == "countdown":
            if 1 <= len(self.latest_detections) <= 5:
                self.countdown_last_face_seen_at = time.monotonic()
            elif len(self.latest_detections) > 5:
                self.mode = "home"
                self.home_message = "参加できる人数は1〜5人です。"
            return

        if self.tracker is None:
            return
        now = time.monotonic()
        if (
            self.face_loss_since is not None
            and now - self.face_loss_since >= FACE_RECOVERY_SECONDS
        ):
            self._return_home("顔を再検出できませんでした。ゲームを終了します。")
            return
        if (
            self.extra_face_since is not None
            and now - self.extra_face_since >= EXTRA_FACE_RETRY_SECONDS
        ):
            self._return_home(
                "新しい顔または追跡の入れ替わりを検出しました。\n"
                "誤判定を防ぐため、ゲームを終了しました。"
            )
            return
        try:
            self.tracker.update(self.latest_detections)
        except ExtraFacesError:
            if self.extra_face_since is None:
                self.extra_face_since = now
            return
        except TrackingError as error:
            self._return_home(
                f"{error}\n誤判定を防ぐため、ゲームを終了しました。"
            )
            return
        if self.extra_face_since is not None:
            paused = now - self.extra_face_since
            if self.face_loss_since is None:
                self.phase_started += paused
                if self.correct_missing_since is not None:
                    self.correct_missing_since += paused
            self.extra_face_since = None
        self._sync_tracking_regions()
        if self.mode in ("question_intro", "question", "answer"):
            required_ids = set(self.tracker.tracks)
        elif self.mode in ("reveal", "explanation", "exit"):
            required_ids = self.correct_ids
        else:
            required_ids = set()
        if required_ids - self.tracker.detected_ids():
            if self.face_loss_since is None:
                self.face_loss_since = now
            return
        if self.face_loss_since is not None:
            paused = now - self.face_loss_since
            self.phase_started += paused
            if self.correct_missing_since is not None:
                self.correct_missing_since += paused
            self.face_loss_since = None
        if self.mode in ("explanation", "exit") and self.correct_ids:
            self._check_correct_presence()

    def _sync_tracking_regions(self) -> None:
        if self.worker is not None:
            boxes = (
                tuple(track.detection.box for track in self.tracker.tracks.values())
                if self.tracker is not None
                else None
            )
            self.worker.set_tracked_boxes(boxes)

    def _start_countdown(self) -> None:
        self.mode = "countdown"
        self.phase_started = time.monotonic()
        self.countdown_last_face_seen_at = self.phase_started
        self.home_message = ""

    def _begin_question(self) -> None:
        if self.tracker is None:
            self._return_home("参加者を追跡できません。")
            return
        self.mode = "question_intro"
        self.phase_started = time.monotonic()

    def _advance_phase(self) -> None:
        now = time.monotonic()
        if self.mode in ("notice", "win", "lose"):
            duration = WIN_SCREEN_SECONDS if self.mode == "win" else RESULT_SCREEN_SECONDS
            if now - self.phase_started >= duration:
                self.home_message = self.end_message if self.mode == "notice" else ""
                self.mode = "home"
            return
        try:
            self.speech.check()
        except RuntimeError as error:
            self._return_home(str(error))
            return

        if self.face_loss_since is not None:
            if now - self.face_loss_since >= FACE_RECOVERY_SECONDS:
                self._return_home("顔を再検出できませんでした。ゲームを終了します。")
                return
            if self.extra_face_since is None:
                return

        if self.extra_face_since is not None:
            if now - self.extra_face_since >= EXTRA_FACE_RETRY_SECONDS:
                self._return_home(
                    "新しい顔または追跡の入れ替わりを検出しました。\n"
                    "誤判定を防ぐため、ゲームを終了しました。"
                )
            return

        if self.mode == "countdown":
            if (
                now - self.countdown_last_face_seen_at
                > COUNTDOWN_FACE_LOSS_GRACE_SECONDS
            ):
                self.mode = "home"
                self.home_message = (
                    "顔が検出されず、カウントダウンを中止しました。"
                )
                return
            if now - self.phase_started >= START_COUNTDOWN_SECONDS:
                if not 1 <= len(self.latest_detections) <= 5:
                    return
                try:
                    self.tracker = FaceTracker(self.latest_detections)
                except TrackingError as error:
                    self.mode = "home"
                    self.home_message = str(error)
                    return
                self._sync_tracking_regions()
                self.current_set_id = self.next_set_id
                self.next_set_id = self.next_set_id % 5 + 1
                self.question_index = 0
                self.mode = "question_intro"
                self._begin_question()

        elif self.mode == "question_intro":
            if now - self.phase_started >= QUESTION_INTRO_SECONDS:
                self.mode = "question"
                self.phase_started = now
                question = self._current_question()
                try:
                    self.speech.speak(f"{question['question']} まるかばつか。")
                except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                    self._return_home(str(error))

        elif self.mode == "question":
            if (
                now - self.phase_started >= 5
                and self.speech.finished()
            ):
                self.mode = "answer"
                self.phase_started = now

        elif self.mode == "answer":
            if now - self.phase_started >= 5:
                if self.tracker is None:
                    self._return_home("参加者を追跡できません。")
                    return
                if (
                    len(self.tracker.detected_ids()) < len(self.tracker.tracks)
                    and now - self.phase_started < 5 + ANSWER_FACE_RETRY_SECONDS
                ):
                    return
                correct_answer = self._current_question()["answer"]
                self.correct_ids, self.wrong_ids = self.tracker.answer(
                    correct_answer
                )
                if not self.tracker.detected_ids():
                    self._return_home(
                        "回答時間終了時に誰も検出できませんでした。未回答として終了します。"
                    )
                    return
                self.mode = "reveal"
                self.phase_started = now

        elif self.mode == "reveal" and now - self.phase_started >= 5:
            self.mode = "explanation"
            self.phase_started = now
            self.correct_missing_since = None
            try:
                self.speech.speak(self._current_question()["explanation"])
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                self._return_home(str(error))
                return

        elif self.mode == "explanation":
            if self.correct_ids:
                self._check_correct_presence()
                if self.mode != "explanation":
                    return
            if (
                now - self.phase_started >= 5
                and self.speech.finished()
            ):
                if self.wrong_ids:
                    self.mode = "exit"
                    self.phase_started = now
                else:
                    self._continue_or_finish()

        elif self.mode == "exit" and now - self.phase_started >= 3:
            if not self.correct_ids:
                self._show_end_screen("lose", "残念！もう一度チャレンジして")
                return
            visible = self.tracker.visible_ids() if self.tracker else set()
            remaining_wrong = visible & self.wrong_ids
            if remaining_wrong:
                self._return_home(
                    "不正解の参加者が退場時間後もカメラ範囲内にいます。"
                    "次の問題へ進まずホーム画面に戻ります。"
                )
                return
            surviving_correct = self.correct_ids & visible
            if not surviving_correct:
                self._return_home(
                    "正解者を確認できなくなりました。誤判定防止のため終了します。"
                )
                return
            self.correct_ids = surviving_correct
            self.tracker.retain(surviving_correct)
            self._sync_tracking_regions()
            self._continue_or_finish()

    def _check_correct_presence(self) -> None:
        if self.tracker is None or not self.correct_ids:
            return
        if self.correct_ids & self.tracker.visible_ids():
            self.correct_missing_since = None
            return
        now = time.monotonic()
        if self.correct_missing_since is None:
            self.correct_missing_since = now
        elif now - self.correct_missing_since >= 1:
            self._return_home(
                "正解者をカメラ内で確認できなくなりました。ゲームを終了します。"
            )

    def _continue_or_finish(self) -> None:
        if not self.correct_ids:
            self._show_end_screen("lose", "残念！もう一度チャレンジして")
            return
        if self.question_index >= 4:
            self._show_end_screen("win", "おめでとう！全問正解！")
            return
        self.question_index += 1
        self._begin_question()

    def _current_question(self) -> dict:
        return self.sets[self.current_set_id][self.question_index]

    def _status_text(self) -> str:
        if self.mode in ("notice", "win", "lose"):
            return self.end_message
        if self.camera_error:
            return f"カメラエラー: {self.camera_error}"
        if self.worker is None:
            return self.camera_error or "カメラを利用できません。"
        _frame, _faces, _sequence, ready, error = self.worker.snapshot()
        if error:
            return f"カメラエラー: {error}"
        if not ready:
            return "カメラを確認しています…"
        if self.face_loss_since is not None:
            remaining = max(
                0,
                math.ceil(
                    FACE_RECOVERY_SECONDS - (time.monotonic() - self.face_loss_since)
                ),
            )
            return f"参加者の顔を再確認中（残り{remaining}秒）"
        if self.extra_face_since is not None:
            return "顔の検出数を確認中です。判定を一時停止しています。"
        if self.mode == "home":
            if self.setup_errors:
                return f"開始できません: {' / '.join(self.setup_errors)}"
            device = self._selected_camera_device()
            camera_name = device.name if device else f"カメラ {self.camera_index}"
            status = (
                f"選択中: {camera_name}"
                "  |  カメラ使用中"
                f"  |  顔検出: {len(self.latest_detections)}人"
            )
            if self.home_message:
                status += f"  |  {self.home_message}"
            return status + "  |  映像は保存・外部送信されません"
        if self.mode == "waiting_for_players":
            return (
                self.home_message
                or "参加者の顔を確認しています。カメラの前に1〜5人でお進みください。"
            )
        if self.mode == "countdown":
            remaining = max(
                0.0, START_COUNTDOWN_SECONDS - (time.monotonic() - self.phase_started)
            )
            if remaining == 0 and not self.latest_detections:
                return (
                    "参加者の顔を確認中（再検出まで最大2秒）"
                    "  |  見つからなければホーム画面に戻ります"
                )
            return (
                f"ゲーム開始まで {max(1, int(remaining + 0.999))} 秒"
                "  |  顔が一時的に見えなくても2秒間待ちます"
            )
        if self.mode == "answer":
            remaining = max(0.0, 5 - (time.monotonic() - self.phase_started))
            if remaining == 0:
                return "回答の判定前に顔を確認しています（最大2秒）"
            return (
                f"回答時間: 残り{int(remaining + 0.999)}秒"
                "  |  カメラ使用中・映像は保存／外部送信しません"
            )
        if self.mode == "reveal":
            return (
                "正解発表中 | まもなく解説へ"
                "  |  カメラ使用中・映像は保存／外部送信しません"
            )
        if self.mode == "explanation":
            return (
                "解説を表示・読み上げ中"
                "  |  カメラ使用中・映像は保存／外部送信しません"
            )
        if self.mode == "exit":
            return (
                "退場時間中 | 不正解の参加者はカメラ範囲から出てください"
                "  |  カメラ使用中・映像は保存／外部送信しません"
            )
        return (
            f"セット {self.current_set_id}  |  "
            f"問題 {self.question_index + 1}/5  |  "
            f"カメラ使用中・映像は保存／外部送信しません"
        )

    def _render(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        self._render_background(width, height)
        if self.face_loss_since is not None:
            self._render_face_recovery(width, height, self.face_loss_since)
        elif self.mode == "home":
            self._panel(width * 0.025, height * 0.04, width * 0.975, height * 0.96)
            self.canvas.create_text(
                width / 2,
                height * 0.17,
                text=APP_TITLE,
                fill=WHITE,
                font=(self.font_family, 60, "bold"),
            )
            self.canvas.create_text(
                width / 2,
                height * 0.36,
                text="カメラの前で左右に移動して〇×に答えよう！",
                fill=PALE_CYAN,
                font=(self.font_family, 24),
            )
            self.canvas.create_text(
                width / 2,
                height * 0.47,
                text="〇  左へ移動                 ×  右へ移動",
                fill=WHITE,
                font=(self.font_family, 26, "bold"),
            )
            self.canvas.create_text(
                width / 2,
                height * 0.70,
                text=(
                    "ホーム画面で顔を1〜5人認識すると、自動で10秒カウントダウン後に出題します。\n"
                    "カウントダウン中に顔を認識できなくなると、カウントダウン前のホーム画面に戻ります。\n"
                    "顔位置を画面に表示するためだけに使い、顔認証・本人特定はしません。\n"
                    "カメラ映像・画像は保存せず、外部へ送信しません。\n"
                    "F11：全画面切替　 Alt+F4：終了"
                ),
                fill=PALE_CYAN,
                font=(self.font_family, 14),
                width=width * 0.90,
                justify="center",
            )
            if self.setup_errors or self.camera_error:
                self.canvas.create_text(
                    width / 2,
                    height * 0.90,
                    text="開始できません。下のスタッフ向け表示を確認してください。",
                    fill=AMBER,
                    font=(self.font_family, 16, "bold"),
                )
        elif self.mode == "camera_test":
            self._render_camera_test(width, height)
        elif self.mode in ("waiting_for_players", "countdown"):
            self._render_home_wait(width, height)
        elif self.mode in ("win", "lose", "notice"):
            self._render_end_screen(width, height)
        elif self.mode == "question_intro":
            self._panel(width * 0.12, height * 0.2, width * 0.88, height * 0.8)
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=f"第{self.question_index + 1}問",
                fill=WHITE,
                font=(self.font_family, 72, "bold"),
            )
        elif self.mode in ("question", "explanation"):
            question = self._current_question()
            self._render_reading(
                width,
                height,
                "問題" if self.mode == "question" else "解説",
                question["question"] if self.mode == "question" else question["explanation"],
            )
        else:
            self._render_game(width, height)

        self.cancel_button.configure(
            state="normal" if self.mode != "home" else "disabled"
        )
        camera_controls_state = "readonly" if self.mode == "home" else "disabled"
        self.camera_combo.configure(state=camera_controls_state)
        self.refresh_camera_button.configure(
            state="normal" if self.mode == "home" else "disabled"
        )
        camera_ready = False
        if self.worker is not None:
            frame, _faces, _sequence, camera_ready, _error = self.worker.snapshot()
            camera_ready = camera_ready and frame is not None
        self.camera_test_button.configure(
            text="カメラ確認を終了  [Esc]"
            if self.mode == "camera_test"
            else "カメラ確認  [C]",
            state=(
                "normal"
                if self.mode == "camera_test"
                or (self.mode == "home" and camera_ready and self.camera_error is None)
                else "disabled"
            ),
        )
        can_start = (
            self.mode in ("home", "waiting_for_players")
            and not self.setup_errors
            and camera_ready
            and self.camera_error is None
        )
        self.start_button.configure(
            text=(
                "参加者待ちを中止"
                if self.mode == "waiting_for_players"
                else "ゲーム開始  [Space]"
            ),
            command=(
                self._cancel_player_wait
                if self.mode == "waiting_for_players"
                else self.start_game
            ),
            state="normal" if can_start else "disabled",
        )

    def _render_background(self, width: int, height: int) -> None:
        if self.background_image is None:
            return
        if self.background_size != (width, height):
            image = self.background_image.resize(
                (width, height), Image.Resampling.LANCZOS
            )
            self.background_photo = ImageTk.PhotoImage(image, master=self.root)
            self.background_size = (width, height)
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")

    def _panel(
        self, left: float, top: float, right: float, bottom: float,
        *, fill: str = PANEL, outline: str = CYAN, border: int = 2,
    ) -> int:
        return self.canvas.create_rectangle(
            left, top, right, bottom, fill=fill, outline=outline, width=border
        )

    def _render_face_recovery(
        self, width: int, height: int, loss_started: float
    ) -> None:
        remaining = max(
            0, math.ceil(FACE_RECOVERY_SECONDS - (time.monotonic() - loss_started))
        )
        self._panel(width * 0.1, height * 0.15, width * 0.9, height * 0.85)
        self.canvas.create_text(
            width / 2, height * 0.3,
            text="参加者の顔を確認しています",
            fill=WHITE, font=(self.font_family, 36, "bold"),
        )
        self.canvas.create_text(
            width / 2, height * 0.5,
            text=str(remaining),
            fill=AMBER, font=(self.font_family, 72, "bold"),
        )
        self.canvas.create_text(
            width / 2, height * 0.7,
            text="カメラの方を向いてお待ちください",
            fill=PALE_CYAN, font=(self.font_family, 24),
        )

    def _render_reading(
        self, width: int, height: int, title: str, text: str
    ) -> None:
        self.canvas.create_text(
            width / 2,
            height * 0.13,
            text=title,
            fill=AMBER,
            font=(self.font_family, 40, "bold"),
        )
        self._panel(width * 0.05, height * 0.22, width * 0.95, height * 0.92)
        size = min(52, max(28, int(width / 25)))
        text_id = self.canvas.create_text(
            width / 2,
            height * 0.55,
            text=text,
            fill=WHITE if title == "問題" else PALE_CYAN,
            font=(self.font_family, size, "bold"),
            width=width * 0.82,
            justify="center",
        )
        bounds = self.canvas.bbox(text_id)
        while bounds and bounds[3] - bounds[1] > height * 0.64 and size > 20:
            size -= 2
            self.canvas.itemconfigure(text_id, font=(self.font_family, size, "bold"))
            bounds = self.canvas.bbox(text_id)

    def _render_end_screen(self, width: int, height: int) -> None:
        self._panel(
            width * 0.07, height * 0.16, width * 0.93, height * 0.84,
            fill=SUBPANEL, outline=MAGENTA if self.mode == "lose" else CYAN,
        )
        self.canvas.create_text(
            width / 2,
            height * 0.46,
            text=self.end_message,
            fill=WHITE,
            font=(self.font_family, 44 if self.mode != "notice" else 30, "bold"),
            width=width - 100,
            justify="center",
        )
        self.canvas.create_text(
            width / 2,
            height * 0.75,
            text="まもなくホーム画面に戻ります",
            fill=PALE_CYAN,
            font=(self.font_family, 20),
        )

    def _render_home_wait(self, width: int, height: int) -> None:
        self._panel(width * 0.025, height * 0.04, width * 0.975, height * 0.96)
        self.canvas.create_text(
            width / 2,
            height * 0.16,
            text=APP_TITLE,
            fill=WHITE,
            font=(self.font_family, 60, "bold"),
        )
        if self.mode == "waiting_for_players":
            message = self.home_message or "参加者の顔を確認しています。"
            detail = "1〜5人でカメラの前にお進みください"
        else:
            remaining = max(
                1, int(START_COUNTDOWN_SECONDS - (time.monotonic() - self.phase_started) + 0.999)
            )
            if time.monotonic() - self.phase_started >= START_COUNTDOWN_SECONDS and not self.latest_detections:
                message = "顔を確認中…"
                detail = "再検出されるとゲームを開始します"
            else:
                message = str(remaining)
                detail = "ゲーム開始まで"
        self.canvas.create_text(
            width / 2,
            height * 0.34,
            text=detail,
            fill=PALE_CYAN,
            font=(self.font_family, 30, "bold"),
        )
        self.canvas.create_text(
            width / 2,
            height * 0.55,
            text=message,
            fill=AMBER if self.mode == "countdown" else PALE_CYAN,
            font=(self.font_family, 64 if self.mode == "countdown" else 30, "bold"),
            width=width - 100,
            justify="center",
        )
        self.canvas.create_text(
            width / 2,
            height * 0.78,
            text=f"検出中の顔: {len(self.latest_detections)}人",
            fill=WHITE,
            font=(self.font_family, 22),
        )
        self._render_faces(
            width, height,
            face_bounds=(width * 0.05, height * 0.57, width * 0.95, height * 0.94),
        )

    def _cancel_player_wait(self) -> None:
        if self.mode != "waiting_for_players":
            return
        self.mode = "home"
        self.home_message = ""
        self._render()

    def _render_camera_test(self, width: int, height: int) -> None:
        self._panel(12, 4, width - 12, 48)
        self.canvas.create_text(
            width / 2,
            28,
            text=(
                f"カメラ確認  |  顔の検出数: {len(self.latest_faces)}人  |  "
                "左右反転  |  映像は保存・送信されません  |  Escで戻る"
            ),
            fill=WHITE,
            font=(self.font_family, 18, "bold"),
        )
        if self.latest_frame is None:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="カメラ映像を待っています…",
                fill=WHITE,
                font=(self.font_family, 24),
            )
            return

        frame_height, frame_width = self.latest_frame.shape[:2]
        available_width = max(width - 24, 1)
        available_height = max(height - 56, 1)
        scale = min(available_width / frame_width, available_height / frame_height)
        display_width = max(1, int(frame_width * scale))
        display_height = max(1, int(frame_height * scale))
        frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(image)
        self._camera_photo = photo
        image_x = (width - display_width) / 2
        image_y = 48 + (available_height - display_height) / 2
        self.canvas.create_image(image_x, image_y, image=photo, anchor="nw")
        for x, y, box_width, box_height in self.latest_faces:
            self.canvas.create_rectangle(
                image_x + x * scale,
                image_y + y * scale,
                image_x + (x + box_width) * scale,
                image_y + (y + box_height) * scale,
                outline=GREEN,
                width=3,
            )

    def _render_game(self, width: int, height: int) -> None:
        question = self._current_question()
        header_top = height * 0.025
        header = self._panel(width * 0.025, header_top, width * 0.975, header_top + 1)
        prompt = self.canvas.create_text(
            width / 2, header_top + 8,
            text=f"★{question['level']}　{question['question']}",
            fill=WHITE, font=(self.font_family, 22, "bold"),
            width=width * 0.86, justify="center", anchor="n",
        )
        prompt_bounds = self.canvas.bbox(prompt)
        prompt_bottom = prompt_bounds[3] if prompt_bounds else height * 0.14
        if self.extra_face_since is not None:
            detail = "顔の検出数を確認中です。判定を一時停止しています。"
        elif self.mode == "reveal":
            detail = f"正解は「{question['answer']}」　解説まで"
        elif self.mode == "exit":
            detail = "不正解の人は3秒以内にカメラの範囲から出てください"
        else:
            detail = "回答中：左右に移動して〇か×を選んでください"
        detail_id = self.canvas.create_text(
            width / 2, prompt_bottom + 4,
            text=detail, fill=AMBER, font=(self.font_family, 18),
            width=width * 0.86, justify="center", anchor="n",
        )
        detail_bounds = self.canvas.bbox(detail_id)
        detail_bottom = detail_bounds[3] if detail_bounds else prompt_bottom + height * 0.08
        header_bottom = detail_bottom + 8
        self.canvas.coords(header, width * 0.025, header_top, width * 0.975, header_bottom)

        zone_top, zone_bottom = header_bottom + 8, height * 0.94
        for left, right, color, symbol in (
            (width * 0.025, width * 0.475, GREEN, "〇"),
            (width * 0.525, width * 0.975, RED, "×"),
        ):
            self._panel(left, zone_top, right, zone_bottom, fill="", outline=color)
            zone_width, zone_height = right - left, zone_bottom - zone_top
            size = max(1, int(min(zone_width, zone_height) * 0.8))
            symbol_id = self.canvas.create_text(
                (left + right) / 2, (zone_top + zone_bottom) / 2,
                text=symbol, fill=color, font=(self.font_family, -size, "bold"),
            )
            bounds = self.canvas.bbox(symbol_id)
            if bounds:
                ratio = min(
                    zone_width * 0.8 / max(1, bounds[2] - bounds[0]),
                    zone_height * 0.8 / max(1, bounds[3] - bounds[1]),
                )
                size = max(1, int(size * ratio))
                self.canvas.itemconfigure(
                    symbol_id, font=(self.font_family, -size, "bold")
                )

        self._render_faces(
            width, height,
            face_bounds=(width * 0.025, zone_top, width * 0.975, zone_bottom),
        )
        display_time = (
            self.extra_face_since
            if self.extra_face_since is not None
            else time.monotonic()
        )
        duration = 3 if self.mode == "exit" else 5
        if self.mode == "answer" and self.extra_face_since is None:
            remaining = max(0, int(duration - (display_time - self.phase_started) + 0.999))
            size = 256
            timer_id = self.canvas.create_text(
                width / 2, height / 2,
                text=str(remaining), fill=AMBER, font=(self.font_family, size, "bold"),
            )
            bounds = self.canvas.bbox(timer_id)
            available_height = max(1, 2 * min(
                height / 2 - header_bottom - 4, zone_bottom - height / 2
            ))
            while bounds and size > 12 and (
                bounds[3] - bounds[1] > available_height
                or bounds[2] - bounds[0] > width * 0.12
            ):
                size -= 4
                self.canvas.itemconfigure(timer_id, font=(self.font_family, size, "bold"))
                bounds = self.canvas.bbox(timer_id)

        elapsed = min(duration, display_time - self.phase_started)
        bar_width = (width - 80) * (1 - elapsed / duration)
        self.canvas.create_rectangle(
            40, height - 24, 40 + bar_width, height - 12, fill=AMBER, outline=""
        )

    def _render_faces(
        self, width: int, height: int,
        face_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        if self.latest_frame is None:
            return
        frame_height, frame_width = self.latest_frame.shape[:2]
        tracked = self.tracker.tracks if self.tracker else {}
        if self.mode in ("countdown", "waiting_for_players"):
            detections = list(enumerate(self.latest_detections, start=1))
        else:
            visible_ids = self.tracker.visible_ids() if self.tracker else set()
            detections = [
                (track_id, track.detection)
                for track_id, track in tracked.items()
                if track_id in visible_ids
            ]

        for track_id, detection in detections:
            x, y, box_width, box_height = detection.box
            left = max(0, int(x * frame_width))
            top = max(0, int(y * frame_height))
            right = min(frame_width, int((x + box_width) * frame_width))
            bottom = min(frame_height, int((y + box_height) * frame_height))
            if right <= left or bottom <= top:
                continue
            crop = self.latest_frame[top:bottom, left:right]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).resize((116, 116), Image.Resampling.LANCZOS)
            mask = Image.new("L", (116, 116), 0)
            ImageDraw.Draw(mask).ellipse((1, 1, 115, 115), fill=255)
            image.putalpha(mask)
            photo = ImageTk.PhotoImage(image)

            fill = GREEN if detection.center_x < 0.5 else RED
            face_label = ""
            if self.mode in ("reveal", "explanation", "exit"):
                if track_id in self.correct_ids:
                    fill = GREEN
                    face_label = "正解！"
                elif track_id in self.wrong_ids:
                    fill = RED
                    face_label = "不正解"
            center_x = detection.center_x * width
            center_y = detection.center_y * height
            if face_bounds is not None:
                left, top, right, bottom = face_bounds
                if detection.center_x < 0.5:
                    center_x = max(left + 64, min(center_x, width * 0.39 - 64))
                else:
                    center_x = min(right - 64, max(center_x, width * 0.61 + 64))
                center_y = max(
                    top + 64, min(center_y, bottom - (100 if face_label else 64))
                )
            self.canvas.create_oval(
                center_x - 64,
                center_y - 64,
                center_x + 64,
                center_y + 64,
                fill=fill,
                outline=WHITE,
                width=5,
            )
            self.canvas.create_image(
                center_x,
                center_y,
                image=photo,
            )
            if face_label:
                self.canvas.create_text(
                    center_x,
                    center_y + 78,
                    text=face_label,
                    fill=WHITE,
                    font=(self.font_family, 18, "bold"),
                )
            if not hasattr(self, "_photo_refs"):
                self._photo_refs = []
            self._photo_refs.append(photo)
        self._photo_refs = self._photo_refs[-5:]

    def _return_home(self, message: str) -> None:
        self._show_end_screen("notice", message)

    def _show_end_screen(self, mode: str, message: str) -> None:
        self.mode = mode
        self.phase_started = time.monotonic()
        self.end_message = message
        self.tracker = None
        self._sync_tracking_regions()
        self.home_message = ""
        self.correct_ids.clear()
        self.wrong_ids.clear()
        self.correct_missing_since = None
        self.extra_face_since = None
        self.face_loss_since = None
        if self.speech is not None:
            self.speech.stop()

    def close(self) -> None:
        if self.speech is not None:
            self.speech.stop()
        if self.worker is not None and not self.worker.stop():
            self._return_home(
                "カメラを解放できませんでした。もう一度終了してください。"
            )
            return
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="使用するカメラ番号（既定値: 0）",
    )
    args = parser.parse_args()
    root = tk.Tk()
    SecurityQuizApp(root, args.camera_index)
    root.mainloop()


if __name__ == "__main__":
    main()
