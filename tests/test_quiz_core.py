import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app import (
    CameraDevice,
    CameraWorker,
    Mp3SoundPlayer,
    SecurityQuizApp,
    choose_japanese_font,
    detect_faces,
    devices_from_names,
    merge_face_boxes,
)
from quiz_core import (
    Detection,
    FaceTracker,
    QuestionDataError,
    TrackingError,
    load_question_sets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_FILE = PROJECT_ROOT / "261113セキュリティクイズ問題.json"


class Mp3SoundPlayerTests(unittest.TestCase):
    def test_stop_and_close_run_on_the_playback_thread(self):
        class FakeMciFunction:
            def __init__(self):
                self.commands = []
                self.play_started = threading.Event()

            def __call__(self, command, output, *_args):
                self.commands.append((command, threading.get_ident()))
                if command.startswith("play "):
                    self.play_started.set()
                if command.startswith("status "):
                    output.value = "playing"
                return 0

        class FakeWinmm:
            def __init__(self):
                self.mciSendStringW = FakeMciFunction()
                self.mciGetErrorStringW = FakeMciFunction()

        winmm = FakeWinmm()
        with patch("app.ctypes.WinDLL", return_value=winmm):
            player = Mp3SoundPlayer(PROJECT_ROOT / "sound")
        try:
            player.play("Timer.mp3")
            self.assertTrue(winmm.mciSendStringW.play_started.wait(2))
        finally:
            player.stop()

        calls = winmm.mciSendStringW.commands
        relevant = [(command, ident) for command, ident in calls
                    if command.startswith(("open ", "play ", "stop ", "close "))]
        self.assertEqual(
            [command.split()[0] for command, _ in relevant],
            ["open", "play", "stop", "close"],
        )
        self.assertEqual(len({ident for _, ident in relevant}), 1)
        self.assertNotEqual(relevant[0][1], threading.get_ident())


class CameraDeviceTests(unittest.TestCase):
    def test_device_names_map_to_stable_directshow_indexes(self):
        devices = devices_from_names([" HD Webcam ", "USB Camera"])
        self.assertEqual(
            devices,
            [
                CameraDevice(0, "HD Webcam"),
                CameraDevice(1, "USB Camera"),
            ],
        )
        self.assertEqual(devices[1].label, "USB Camera（カメラ 1）")

    def test_face_boxes_from_different_detectors_are_deduplicated(self):
        boxes = merge_face_boxes(
            [(100, 100, 80, 80), (300, 100, 80, 80)],
            [(105, 102, 82, 82), (500, 100, 80, 80)],
        )

        self.assertEqual(
            boxes,
            [(100, 100, 80, 80), (300, 100, 80, 80), (500, 100, 80, 80)],
        )

    def test_empty_device_names_are_ignored(self):
        self.assertEqual(devices_from_names(["", "  ", "Camera"]), [
            CameraDevice(2, "Camera"),
        ])

    def test_quiz_search_excludes_faces_outside_tracked_regions(self):
        image = np.zeros((300, 1000), dtype=np.uint8)
        image[120:160, 180:220] = 1
        image[120:160, 800:840] = 1

        def find_faces(region, **_kwargs):
            ys, xs = np.where(region == 1)
            boxes = []
            columns = np.unique(xs)
            for group in np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1):
                if not len(group):
                    continue
                mask = (xs >= group[0]) & (xs <= group[-1])
                boxes.append((int(xs[mask].min()), int(ys[mask].min()), 40, 40))
            return boxes

        frontal = Mock()
        frontal.detectMultiScale.side_effect = find_faces
        profile = Mock()
        profile.detectMultiScale.return_value = []
        full = detect_faces(image, frontal, profile, None)
        tracked = detect_faces(image, frontal, profile, ((0.18, 0.4, 0.04, 0.133),))

        self.assertEqual(full, [(180, 120, 40, 40), (800, 120, 40, 40)])
        self.assertEqual(tracked, [(180, 120, 40, 40)])

    def test_profile_coordinates_are_restored_from_flipped_region(self):
        image = np.zeros((300, 1000), dtype=np.uint8)
        frontal = Mock()
        frontal.detectMultiScale.return_value = []
        profile = Mock()
        profile.detectMultiScale.side_effect = [[], [(90, 10, 40, 40)]]

        boxes = detect_faces(image, frontal, profile, ((0.18, 0.4, 0.04, 0.133),))

        self.assertEqual(boxes, [(320, 70, 40, 40)])

    def test_scoped_detection_retries_low_contrast_near_previous_face(self):
        image = np.tile(np.arange(1000, dtype=np.uint16) % 40 + 80, (300, 1)).astype(np.uint8)
        frontal = Mock()
        frontal.detectMultiScale.side_effect = [
            [],
            [(180, 60, 40, 40), (320, 60, 40, 40), (131, 30, 100, 100)],
        ]
        profile = Mock()
        profile.detectMultiScale.return_value = []
        tracked = ((0.18, 0.4, 0.04, 0.1333333333),)

        boxes = detect_faces(image, frontal, profile, tracked)

        self.assertEqual(boxes, [(180, 120, 40, 40)])
        self.assertEqual(frontal.detectMultiScale.call_count, 2)
        original = frontal.detectMultiScale.call_args_list[0].args[0]
        enhanced = frontal.detectMultiScale.call_args_list[1].args[0]
        self.assertEqual(original.shape, enhanced.shape)
        self.assertGreater(int(enhanced.max()) - int(enhanced.min()),
                           int(original.max()) - int(original.min()))

    def test_home_detection_does_not_retry_with_enhanced_image(self):
        image = np.zeros((300, 1000), dtype=np.uint8)
        frontal = Mock()
        frontal.detectMultiScale.return_value = []
        profile = Mock()
        profile.detectMultiScale.return_value = []

        self.assertEqual(detect_faces(image, frontal, profile, None), [])
        self.assertEqual(frontal.detectMultiScale.call_count, 1)

    def test_scope_change_invalidates_old_full_frame_detections(self):
        worker = CameraWorker(0)
        worker.faces = [(800, 120, 40, 40)]
        worker.sequence = 4
        worker.set_tracked_boxes(((0.18, 0.4, 0.04, 0.133),))
        self.assertEqual(worker.faces, [])
        self.assertEqual(worker.sequence, 5)
        worker.faces = [(180, 120, 40, 40)]
        worker.set_tracked_boxes(None)
        self.assertEqual(worker.faces, [])
        self.assertEqual(worker.sequence, 6)

    def test_scoped_detection_searches_each_participant_near_image_edges(self):
        image = np.zeros((300, 1000), dtype=np.uint8)
        frontal = Mock()
        frontal.detectMultiScale.side_effect = [
            [(5, 10, 40, 40)],
            [(90, 10, 40, 40)],
        ]
        profile = Mock()
        profile.detectMultiScale.return_value = []

        boxes = detect_faces(
            image,
            frontal,
            profile,
            ((0.0, 0.0, 0.04, 0.133), (0.96, 0.0, 0.04, 0.133)),
        )

        self.assertEqual(boxes, [(5, 10, 40, 40), (820, 10, 40, 40)])

    def test_scoped_detection_finds_small_face_after_a_larger_move(self):
        image = np.zeros((300, 1000), dtype=np.uint8)
        image[120:156, 390:426] = 1

        def find_faces(region, **kwargs):
            ys, xs = np.where(region == 1)
            if not len(xs) or kwargs["minSize"][0] > 36 or kwargs["minNeighbors"] > 4:
                return []
            if xs.max() - xs.min() + 1 < 36:
                return []
            return [(int(xs.min()), int(ys.min()), 36, 36)]

        frontal = Mock()
        frontal.detectMultiScale.side_effect = find_faces
        profile = Mock()
        profile.detectMultiScale.return_value = []
        boxes = detect_faces(image, frontal, profile, ((0.18, 0.4, 0.04, 0.133),))
        self.assertEqual(boxes, [(390, 120, 36, 36)])


class QuestionDataTests(unittest.TestCase):
    def test_shipped_questions_have_five_complete_sets(self):
        sets = load_question_sets(QUESTION_FILE)
        self.assertEqual(list(sets), [1, 2, 3, 4, 5])
        self.assertTrue(all(len(questions) == 5 for questions in sets.values()))
        self.assertTrue(
            all(
                [question["level"] for question in questions] == [1, 2, 3, 4, 5]
                for questions in sets.values()
            )
        )

    def test_invalid_answer_is_rejected(self):
        data = json.loads(QUESTION_FILE.read_text(encoding="utf-8-sig"))
        data["sets"][0]["questions"][0]["answer"] = "〇か×"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(QuestionDataError):
                load_question_sets(path)

    def test_invalid_set_number_is_rejected(self):
        data = json.loads(QUESTION_FILE.read_text(encoding="utf-8-sig"))
        data["sets"][0]["set_id"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(QuestionDataError):
                load_question_sets(path)


class FaceTrackerTests(unittest.TestCase):
    @staticmethod
    def face(x, y=0.5):
        return Detection(x, y, (x - 0.04, y - 0.04, 0.08, 0.08))

    def test_answers_use_mirrored_face_center(self):
        tracker = FaceTracker([self.face(0.2), self.face(0.8)])
        tracker.update([self.face(0.25), self.face(0.75)])
        correct, wrong = tracker.answer("〇")
        self.assertEqual(correct, {1})
        self.assertEqual(wrong, {2})

    def test_cross_answer_uses_the_right_side(self):
        tracker = FaceTracker([self.face(0.2), self.face(0.8)])
        correct, wrong = tracker.answer("×")
        self.assertEqual(correct, {2})
        self.assertEqual(wrong, {1})

    def test_no_visible_face_is_distinct_from_wrong_answers(self):
        with patch("quiz_core.time.monotonic", return_value=10):
            tracker = FaceTracker([self.face(0.2)])
        with patch("quiz_core.time.monotonic", return_value=13.1):
            tracker.update([])
            correct, wrong = tracker.answer("〇")
            self.assertEqual(correct, set())
            self.assertEqual(wrong, {1})
            self.assertEqual(tracker.visible_ids(), set())

    def test_brief_detection_drop_is_visible_but_not_used_for_answer(self):
        tracker = FaceTracker([self.face(0.2)])
        tracker.update([])
        correct, wrong = tracker.answer("〇")
        self.assertEqual(correct, set())
        self.assertEqual(wrong, {1})
        self.assertEqual(tracker.visible_ids(), {1})

    def test_detection_drop_grace_is_three_seconds_not_frame_count(self):
        with patch("quiz_core.time.monotonic", return_value=10):
            tracker = FaceTracker([self.face(0.2)])
        with patch("quiz_core.time.monotonic", return_value=12.9):
            for _ in range(100):
                tracker.update([])
            self.assertEqual(tracker.visible_ids(), {1})
            tracker.update([self.face(0.21)])
            self.assertEqual(tracker.detected_ids(), {1})

        with patch("quiz_core.time.monotonic", return_value=16):
            tracker.update([])
            self.assertEqual(tracker.visible_ids(), set())

    def test_extra_faces_pause_but_ambiguous_positions_are_accepted(self):
        tracker = FaceTracker([self.face(0.2), self.face(0.8)])
        with self.assertRaises(TrackingError):
            tracker.update([self.face(0.2), self.face(0.5), self.face(0.8)])

        tracker = FaceTracker([self.face(0.45), self.face(0.55)])
        tracker.update([self.face(0.5), self.face(0.501)])
        self.assertEqual(tracker.detected_ids(), {1, 2})

    def test_lost_tracks_can_be_removed_after_exit(self):
        tracker = FaceTracker([self.face(0.2), self.face(0.8)])
        tracker.update([self.face(0.2)])
        tracker.retain({1})
        self.assertEqual(set(tracker.tracks), {1})

    def test_long_lost_track_can_be_reassociated(self):
        with patch("quiz_core.time.monotonic", return_value=10):
            tracker = FaceTracker([self.face(0.2)])
        with patch("quiz_core.time.monotonic", return_value=13.1):
            tracker.update([])
        with patch("quiz_core.time.monotonic", return_value=23.1):
            tracker.update([self.face(0.2)])
            self.assertEqual(tracker.detected_ids(), {1})

    def test_face_can_be_reassociated_after_three_but_before_ten_seconds(self):
        with patch("quiz_core.time.monotonic", return_value=10):
            tracker = FaceTracker([self.face(0.2)])
        with patch("quiz_core.time.monotonic", return_value=11):
            tracker.update([])
        with patch("quiz_core.time.monotonic", return_value=20.9):
            self.assertEqual(tracker.visible_ids(), set())
            tracker.update([self.face(0.21)])
            self.assertEqual(tracker.detected_ids(), {1})

    def test_normal_answer_movement_does_not_abort_tracking(self):
        tracker = FaceTracker([self.face(0.5)])
        tracker.update([self.face(0.2)])
        correct, wrong = tracker.answer("〇")
        self.assertEqual(correct, {1})
        self.assertEqual(wrong, set())


class GameProgressTests(unittest.TestCase):
    def make_app(self, mode, started_at):
        app = SecurityQuizApp.__new__(SecurityQuizApp)
        app.mode = mode
        app.phase_started = started_at
        app.current_set_id = 1
        app.question_index = 0
        app.sets = {
            1: [
                {
                    "answer": "〇",
                    "question": "問題テキスト",
                    "explanation": "解説テキスト",
                }
            ]
        }
        app.speech = Mock()
        app.speech.finished.return_value = True
        app.sound_effects = Mock()
        app.tracker = Mock()
        app.worker = None
        app.tracker.tracks = {1: Mock(), 2: Mock()}
        app.tracker.visible_ids.return_value = {1, 2}
        app.tracker.detected_ids.return_value = {1, 2}
        app.tracker.answer.return_value = ({1}, {2})
        app.correct_ids = {1}
        app.wrong_ids = {2}
        app.correct_missing_since = None
        app.extra_face_since = None
        app.face_loss_since = None
        app.font_family = "Yu Gothic UI"
        app.background_image = None
        app.end_message = ""
        app._return_home = Mock()
        app._continue_or_finish = Mock()
        return app

    def test_escape_closes_application_from_game_and_home(self):
        for mode in ("home", "countdown", "answer", "explanation"):
            with self.subTest(mode=mode):
                app = self.make_app(mode, 10)
                app.close = Mock()
                self.assertEqual(app._escape_pressed(None), "break")
                app.close.assert_called_once_with()

    def test_escape_returns_from_camera_test_without_exiting(self):
        app = self.make_app("camera_test", 10)
        app.close = Mock()
        app._render = Mock()
        self.assertEqual(app._escape_pressed(None), "break")
        self.assertEqual(app.mode, "home")
        app._render.assert_called_once_with()
        app.close.assert_not_called()

    def test_answer_timeout_shows_result(self):
        app = self.make_app("answer", 10)
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")
        self.assertEqual(app.correct_ids, {1})
        self.assertEqual(app.wrong_ids, {2})

    def test_question_intro_plays_numbered_cues_in_order(self):
        app = self.make_app("question_intro", 10)
        for index in range(5):
            app.question_index = index
            app._begin_question()
            app.sound_effects.play.assert_called_with(
                "Q_sound.mp3", f"Q{index + 1}.mp3"
            )
        self.assertEqual(app.sound_effects.play.call_count, 5)

    def test_answer_timer_starts_on_entry_and_reveal_replaces_it(self):
        for correct_ids, expected in (
            ({1}, "CorrectAnswer.mp3"),
            (set(), "IncorrectAnswer.mp3"),
        ):
            with self.subTest(correct_ids=correct_ids):
                app = self.make_app("answer", 10)
                app.tracker.detected_ids.return_value = {1, 2}
                app.tracker.answer.return_value = (correct_ids, {2})
                app._advance_phase = SecurityQuizApp._advance_phase.__get__(app)
                with patch("app.time.monotonic", return_value=15):
                    app.mode = "question"
                    app._advance_phase()
                app.sound_effects.play.assert_called_once_with("Timer.mp3")
                app.sound_effects.reset_mock()
                with patch("app.time.monotonic", return_value=21):
                    app._advance_phase()
                app.sound_effects.stop.assert_called_once_with()
                app.sound_effects.play.assert_called_once_with(expected)
                self.assertEqual(app.mode, "reveal")

    def test_staff_can_choose_lose_screen(self):
        app = self.make_app("explanation", 10)
        app.correct_ids = set()
        app.wrong_ids = set()
        with patch("app.time.monotonic", return_value=16):
            app._advance_phase()
        self.assertEqual(app.mode, "explanation")
        app._staff_lose()
        self.assertEqual(app.mode, "lose")
        self.assertEqual(app.end_message, "残念！もう一度チャレンジして")
        app.sound_effects.play.assert_called_once_with("End_sound.mp3")

        app.sound_effects.reset_mock()
        app._show_end_screen("win", "おめでとう！全問正解！")
        app.sound_effects.play.assert_called_once_with("CorrectAnswerFinal.mp3")
        app.sound_effects.reset_mock()
        app._show_end_screen("lose", "残念！もう一度チャレンジして")
        app.sound_effects.play.assert_called_once_with("End_sound.mp3")

    def test_extra_face_does_not_pause_and_does_not_guess_result(self):
        app = self.make_app("answer", 6)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = [
            FaceTrackerTests.face(0.2),
            FaceTrackerTests.face(0.8),
        ]
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        self.assertEqual(app.extra_face_since, 11)
        app._return_home.assert_not_called()
        with patch("app.time.monotonic", return_value=11.5):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")
        self.assertEqual(app.correct_ids, set())
        self.assertEqual(app.wrong_ids, set())
        app.sound_effects.play.assert_not_called()

    def test_question_intro_waits_four_seconds_before_speaking_for_each_question(self):
        app = self.make_app("answer", 10)
        app.sets[1].append(dict(app.sets[1][0]))
        for index in (0, 1):
            app.question_index = index
            app.speech.speak.reset_mock()
            with patch("app.time.monotonic", return_value=20):
                app._begin_question()
            self.assertEqual(app.mode, "question_intro")
            with patch("app.time.monotonic", return_value=23.9):
                app._advance_phase()
            app.speech.speak.assert_not_called()
            with patch("app.time.monotonic", return_value=24):
                app._advance_phase()
            self.assertEqual(app.mode, "question")
            app.speech.speak.assert_called_once_with("問題テキスト まるかばつか。")

    def test_intro_screen_is_rendered(self):
        app = self.make_app("question_intro", 10)
        app.question_index = 1
        app.canvas = Mock()
        app.canvas.winfo_width.return_value = 1280
        app.canvas.winfo_height.return_value = 760
        app.cancel_button = Mock()
        app.camera_combo = Mock()
        app.refresh_camera_button = Mock()
        app.camera_test_button = Mock()
        app.start_button = Mock()
        app.setup_errors = []
        app.camera_error = None

        app._render()
        labels = [call.kwargs.get("text") for call in app.canvas.create_text.call_args_list]
        self.assertIn("第2問", labels)

    def test_background_is_reused_until_canvas_size_changes(self):
        app = self.make_app("home", 10)
        app.canvas = Mock()
        app.root = Mock()
        app.background_image = Mock()
        app.background_size = None
        app.background_photo = None
        with patch("app.ImageTk.PhotoImage") as photo:
            app._render_background(960, 500)
            app._render_background(960, 500)
            app._render_background(1280, 660)
        self.assertEqual(app.background_image.resize.call_count, 2)
        self.assertEqual(photo.call_count, 2)
        self.assertEqual(app.background_size, (1280, 660))
        self.assertEqual(app.canvas.create_image.call_count, 3)

    def test_f11_toggles_fullscreen_without_changing_game_mode(self):
        app = self.make_app("answer", 10)
        app.root = Mock()
        app.root.attributes.side_effect = [1, None, 0, None]
        self.assertEqual(app._toggle_fullscreen(Mock()), "break")
        self.assertEqual(app.mode, "answer")
        self.assertEqual(app._toggle_fullscreen(Mock()), "break")
        self.assertEqual(
            app.root.attributes.call_args_list,
            [
                unittest.mock.call("-fullscreen"),
                unittest.mock.call("-fullscreen", False),
                unittest.mock.call("-fullscreen"),
                unittest.mock.call("-fullscreen", True),
            ],
        )

    def test_home_title_and_panel_use_requested_style(self):
        app = self.make_app("home", 10)
        app.canvas = Mock()
        app.canvas.winfo_width.return_value = 1280
        app.canvas.winfo_height.return_value = 660
        app.cancel_button = Mock()
        app.camera_combo = Mock()
        app.refresh_camera_button = Mock()
        app.camera_test_button = Mock()
        app.start_button = Mock()
        app.setup_errors = []
        app.camera_error = None
        app._render()
        title = app.canvas.create_text.call_args_list[0].kwargs
        self.assertEqual(title["font"][1], 60)
        self.assertEqual(title["fill"], "#F0F9FF")
        panel = app.canvas.create_rectangle.call_args_list[0].kwargs
        self.assertEqual((panel["fill"], panel["outline"]), ("#0B1D26", "#00E5FF"))

    def test_next_question_also_starts_with_intro(self):
        app = self.make_app("explanation", 10)
        app._continue_or_finish = SecurityQuizApp._continue_or_finish.__get__(app)
        with patch("app.time.monotonic", return_value=20):
            app._continue_or_finish()
        self.assertEqual(app.question_index, 1)
        self.assertEqual(app.mode, "question_intro")
        self.assertEqual(app.phase_started, 20)

    def test_japanese_font_selection_uses_installed_fallback(self):
        self.assertEqual(choose_japanese_font(("Yu Gothic UI",)), "Yu Gothic UI")
        self.assertEqual(choose_japanese_font(("Meiryo UI",)), "Meiryo UI")
        self.assertEqual(
            choose_japanese_font(("BIZ UDPゴシック", "Yu Gothic UI")),
            "BIZ UDPゴシック",
        )
        self.assertEqual(
            choose_japanese_font(("Yu Gothic UI", "Noto Sans JP")), "Noto Sans JP"
        )

    def test_missing_face_does_not_pause_answer(self):
        app = self.make_app("answer", 8)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = []
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        self.assertEqual(app.face_loss_since, 11)
        with patch("app.time.monotonic", return_value=20.9):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")
        self.assertEqual((app.correct_ids, app.wrong_ids), (set(), set()))
        app._return_home.assert_not_called()

    def test_missing_face_waits_without_timeout_and_accepts_late_recovery(self):
        for process_frame in (False, True):
            with self.subTest(process_frame=process_frame):
                app = self.make_app("question", 8)
                with patch("app.time.monotonic", return_value=10):
                    app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
                app.latest_detections = []
                with patch("app.time.monotonic", return_value=11):
                    app._process_new_frame()
                app.latest_detections = [FaceTrackerTests.face(0.2)]
                with patch("app.time.monotonic", return_value=121):
                    if process_frame:
                        app._process_new_frame()
                    else:
                        app._advance_phase()
                app._return_home.assert_not_called()
                if process_frame:
                    self.assertIsNone(app.face_loss_since)
                else:
                    self.assertEqual(app.face_loss_since, 11)

    def test_missing_player_pauses_explanation(self):
        app = self.make_app("answer", 8)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([
                FaceTrackerTests.face(0.2), FaceTrackerTests.face(0.8)
            ])
        app.latest_detections = [FaceTrackerTests.face(0.2)]
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        self.assertEqual(app.face_loss_since, 11)

        app.mode = "explanation"
        app.face_loss_since = None
        with patch("app.time.monotonic", return_value=12):
            app._process_new_frame()
        self.assertEqual(app.face_loss_since, 12)

    def test_detection_count_changes_do_not_shift_timer(self):
        app = self.make_app("answer", 6)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = []
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        app.latest_detections = [
            FaceTrackerTests.face(0.2), FaceTrackerTests.face(0.8)
        ]
        with patch("app.time.monotonic", return_value=12):
            app._process_new_frame()
        app.latest_detections = [FaceTrackerTests.face(0.2)]
        with patch("app.time.monotonic", return_value=13):
            app._process_new_frame()
        self.assertEqual(app.phase_started, 6)
        self.assertIsNone(app.face_loss_since)
        self.assertIsNone(app.extra_face_since)

    def test_large_position_change_resumes_when_count_matches(self):
        app = self.make_app("question", 8)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = []
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        app.latest_detections = [FaceTrackerTests.face(0.8)]
        with patch("app.time.monotonic", return_value=15):
            app._process_new_frame()
        app._return_home.assert_not_called()
        self.assertIsNone(app.face_loss_since)
        self.assertEqual(app.tracker.tracks[1].detection.center_x, 0.8)

    def test_persistent_extra_face_progresses_without_judging(self):
        app = self.make_app("answer", 6)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = [
            FaceTrackerTests.face(0.2),
            FaceTrackerTests.face(0.8),
        ]
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        with patch("app.time.monotonic", return_value=113):
            app._advance_phase()
        app._return_home.assert_not_called()
        self.assertEqual(app.extra_face_since, 11)
        self.assertEqual(app.mode, "reveal")
        self.assertEqual((app.correct_ids, app.wrong_ids), (set(), set()))

    def test_extra_face_accepts_late_recovery(self):
        app = self.make_app("answer", 6)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = [
            FaceTrackerTests.face(0.2),
            FaceTrackerTests.face(0.8),
        ]
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        app.latest_detections = [FaceTrackerTests.face(0.2)]
        with patch("app.time.monotonic", return_value=113):
            app._process_new_frame()
        app._return_home.assert_not_called()
        self.assertIsNone(app.face_loss_since)

    def test_answer_uses_recovered_detection_at_deadline(self):
        app = self.make_app("answer", 10)
        app.tracker.detected_ids.return_value = {1}
        with patch("app.time.monotonic", return_value=14):
            app._advance_phase()
        self.assertEqual(app.mode, "answer")
        app.tracker.answer.assert_not_called()

        app.tracker.detected_ids.return_value = {1, 2}
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")
        app.tracker.answer.assert_called_once_with("〇")

    def test_answer_progresses_without_using_stale_faces(self):
        app = self.make_app("answer", 10)
        app.tracker.detected_ids.return_value = set()
        with patch("app.time.monotonic", return_value=17):
            app._advance_phase()
        app._return_home.assert_not_called()
        app.tracker.answer.assert_not_called()
        self.assertEqual(app.mode, "reveal")
        self.assertEqual((app.correct_ids, app.wrong_ids), (set(), set()))
        app.sound_effects.play.assert_not_called()

    def test_staff_start_waits_for_face_without_consuming_set(self):
        app = SecurityQuizApp.__new__(SecurityQuizApp)
        app.mode = "home"
        app.setup_errors = []
        app.camera_error = None
        app.worker = Mock()
        app.worker.snapshot.return_value = (object(), [], 1, True, None)
        app.latest_detections = []
        app.tracker = None
        app.home_message = ""
        app.next_set_id = 4
        app._render = Mock()

        app.start_game()

        self.assertEqual(app.mode, "waiting_for_players")
        self.assertEqual(app.next_set_id, 4)
        self.assertIn("顔を確認", app.home_message)

    def test_staff_can_start_without_detected_faces(self):
        app = self.make_countdown_app([])
        app.mode = "waiting_for_players"
        app._staff_start()
        self.assertIsNone(app.tracker)
        self.assertEqual(app.current_set_id, 3)
        self.assertEqual(app.next_set_id, 4)
        self.assertEqual(app.question_index, 0)
        app._begin_question.assert_called_once_with()

    def test_manual_game_without_tracker_progresses_without_judging(self):
        app = self.make_app("answer", 10)
        app.tracker = None
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")
        self.assertEqual((app.correct_ids, app.wrong_ids), (set(), set()))
        app.sound_effects.stop.assert_called_once_with()
        app.sound_effects.play.assert_not_called()
        with patch("app.time.monotonic", return_value=20):
            app._advance_phase()
        self.assertEqual(app.mode, "explanation")
        app.camera_error = None
        app.worker = Mock()
        app.worker.snapshot.return_value = (object(), [], 1, True, None)
        self.assertIn("正誤を確認できませんでした", app._status_text())
        self.assertIn("F8", app._status_text())
        self.assertIn("F9", app._status_text())

    def test_staff_keys_only_work_on_explanation(self):
        app = self.make_app("answer", 10)
        app._show_end_screen = Mock()
        app._staff_next()
        app._staff_lose()
        app._continue_or_finish.assert_not_called()
        app._show_end_screen.assert_not_called()
        app.speech.stop.assert_not_called()

    def test_staff_next_overrides_result_and_finishes_final_question(self):
        for index in (0, 4):
            with self.subTest(index=index):
                app = self.make_app("explanation", 10)
                app.question_index = index
                app.correct_ids = set()
                app._continue_or_finish = SecurityQuizApp._continue_or_finish.__get__(app)
                app._begin_question = Mock()
                app._show_end_screen = Mock()
                app._staff_next()
                app.speech.stop.assert_called_once_with()
                if index == 0:
                    self.assertEqual(app.question_index, 1)
                    app._begin_question.assert_called_once_with()
                    app._show_end_screen.assert_not_called()
                else:
                    app._show_end_screen.assert_called_once_with(
                        "win", "おめでとう！全問正解！"
                    )

    def test_update_dispatches_answer_phase_timer(self):
        app = SecurityQuizApp.__new__(SecurityQuizApp)
        app.worker = None
        app.mode = "answer"
        app._advance_phase = Mock()
        app._status_text = Mock(return_value="回答中")
        app.status_var = Mock()
        app._render = Mock()
        app.root = Mock()

        app._update()

        app._advance_phase.assert_called_once_with()
        app.root.after.assert_called_once()

    def test_update_dispatches_countdown_timer(self):
        app = SecurityQuizApp.__new__(SecurityQuizApp)
        app.worker = None
        app.mode = "countdown"
        app.extra_face_since = None
        app._advance_phase = Mock()
        app._status_text = Mock(return_value="ゲーム開始まで 5 秒")
        app.status_var = Mock()
        app._render = Mock()
        app.root = Mock()

        app._update()

        app._advance_phase.assert_called_once_with()
        app.root.after.assert_called_once()

    def test_result_screen_advances_to_explanation(self):
        app = self.make_app("reveal", 10)
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "explanation")
        app.speech.speak.assert_called_once_with("解説テキスト")

    def test_explanation_automatically_ends_when_anyone_is_wrong(self):
        app = self.make_app("explanation", 10)
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "lose")
        app._continue_or_finish.assert_not_called()

    def test_explanation_automatically_continues_when_everyone_is_correct(self):
        app = self.make_app("explanation", 10)
        app.wrong_ids = set()
        with patch("app.time.monotonic", return_value=14.9):
            app._advance_phase()
        app._continue_or_finish.assert_not_called()
        app.speech.finished.return_value = False
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "explanation")
        app._continue_or_finish.assert_not_called()
        app.speech.finished.return_value = True
        with patch("app.time.monotonic", return_value=16):
            app._advance_phase()
        app._continue_or_finish.assert_called_once_with()

    def test_unverified_answer_waits_for_staff_even_after_face_recovery(self):
        app = self.make_app("answer", 10)
        app.tracker.detected_ids.return_value = {1}
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        app.tracker.detected_ids.return_value = {1, 2}
        with patch("app.time.monotonic", return_value=20):
            app._advance_phase()
        with patch("app.time.monotonic", return_value=1000):
            app._advance_phase()
        self.assertEqual(app.mode, "explanation")
        app._continue_or_finish.assert_not_called()
        app._staff_next()
        app.speech.stop.assert_called_once_with()
        app._continue_or_finish.assert_called_once_with()

    def test_final_correct_answer_automatically_wins_even_if_face_disappears_later(self):
        app = self.make_app("answer", 10)
        app.tracker.answer.return_value = ({1, 2}, set())
        app._continue_or_finish = SecurityQuizApp._continue_or_finish.__get__(app)
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        app.face_loss_since = 16
        with patch("app.time.monotonic", return_value=20):
            app._advance_phase()
        app.question_index = 4
        with patch("app.time.monotonic", return_value=25):
            app._advance_phase()
        self.assertEqual(app.mode, "win")
        app.sound_effects.play.assert_called_with("CorrectAnswerFinal.mp3")

    def make_countdown_app(self, detections):
        app = SecurityQuizApp.__new__(SecurityQuizApp)
        app.mode = "countdown"
        app.extra_face_since = None
        app.face_loss_since = None
        app.font_family = "Yu Gothic UI"
        app.phase_started = 10
        app.countdown_last_face_seen_at = 20 if detections else 10
        app.home_message = ""
        app.setup_errors = []
        app.camera_error = None
        app.latest_detections = detections
        app.speech = Mock()
        app.tracker = None
        app.worker = None
        app.next_set_id = 3
        app.current_set_id = 1
        app.question_index = 4
        app._begin_question = Mock()
        return app

    def test_waiting_for_one_face_starts_countdown(self):
        app = self.make_countdown_app([FaceTrackerTests.face(0.5)])
        app.mode = "waiting_for_players"

        with patch("app.time.monotonic", return_value=20):
            app._process_new_frame()

        self.assertEqual(app.mode, "countdown")
        self.assertEqual(app.phase_started, 20)

    def test_home_starts_countdown_automatically_when_face_is_detected(self):
        app = self.make_countdown_app([FaceTrackerTests.face(0.5)])
        app.mode = "home"

        with patch("app.time.monotonic", return_value=20):
            app._process_new_frame()

        self.assertEqual(app.mode, "countdown")
        self.assertEqual(app.phase_started, 20)

    def test_home_does_not_start_countdown_without_face(self):
        app = self.make_countdown_app([])
        app.mode = "home"

        app._process_new_frame()

        self.assertEqual(app.mode, "home")

    def test_home_does_not_start_when_setup_has_errors(self):
        app = self.make_countdown_app([FaceTrackerTests.face(0.5)])
        app.mode = "home"
        app.setup_errors = ["音声が利用できません"]

        app._process_new_frame()

        self.assertEqual(app.mode, "home")

    def test_home_countdown_displays_ten_seconds(self):
        app = self.make_countdown_app([FaceTrackerTests.face(0.5)])
        app.canvas = Mock()
        app._render_faces = Mock()

        with patch("app.time.monotonic", return_value=10):
            app._render_home_wait(1280, 760)

        labels = [call.kwargs.get("text") for call in app.canvas.create_text.call_args_list]
        self.assertIn("10", labels)
        self.assertIn("セキュリティ〇×クイズ", labels)

    def test_countdown_expiry_starts_only_after_ten_seconds(self):
        app = self.make_countdown_app([FaceTrackerTests.face(0.5)])

        with patch("app.time.monotonic", return_value=19.9):
            app._advance_phase()

        self.assertEqual(app.mode, "countdown")
        app._begin_question.assert_not_called()

    def test_countdown_returns_home_when_no_face_is_detected(self):
        app = self.make_countdown_app([])

        app._process_new_frame()

        self.assertEqual(app.mode, "countdown")

        with patch("app.time.monotonic", return_value=12.1):
            app._advance_phase()

        self.assertEqual(app.mode, "home")
        self.assertIn("顔が検出されず", app.home_message)

    def test_countdown_waits_for_face_to_reappear_within_grace(self):
        app = self.make_countdown_app([])
        app.phase_started = 5
        app.countdown_last_face_seen_at = 15

        with patch("app.time.monotonic", return_value=15.5):
            app._process_new_frame()
            app._advance_phase()

        self.assertEqual(app.mode, "countdown")

        app.latest_detections = [FaceTrackerTests.face(0.5)]
        with patch("app.time.monotonic", return_value=15.6):
            app._process_new_frame()
            app._advance_phase()

        self.assertEqual(app.mode, "question_intro")

    def test_countdown_rejects_more_than_five_faces(self):
        app = self.make_countdown_app(
            [FaceTrackerTests.face(x) for x in (0.1, 0.25, 0.4, 0.6, 0.75, 0.9)]
        )

        app._process_new_frame()

        self.assertEqual(app.mode, "home")
        self.assertIn("1〜5人", app.home_message)

    def test_countdown_expiry_starts_game_and_advances_set(self):
        app = self.make_countdown_app([FaceTrackerTests.face(0.3)])
        app.worker = Mock()

        with patch("app.time.monotonic", return_value=20):
            app._advance_phase()
            self.assertEqual(app.tracker.visible_ids(), {1})

        self.assertEqual(app.mode, "question_intro")
        self.assertEqual(app.current_set_id, 3)
        self.assertEqual(app.next_set_id, 4)
        self.assertEqual(app.question_index, 0)
        app.worker.set_tracked_boxes.assert_called_once_with(
            (FaceTrackerTests.face(0.3).box,)
        )
        app._begin_question.assert_called_once_with()

    def test_search_regions_follow_participant_movement(self):
        app = self.make_app("answer", 10)
        app.worker = Mock()
        app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = [FaceTrackerTests.face(0.3)]

        app._process_new_frame()

        app.worker.set_tracked_boxes.assert_called_once_with(
            (FaceTrackerTests.face(0.3).box,)
        )

    def test_return_home_restores_full_frame_search(self):
        app = self.make_app("answer", 10)
        app.worker = Mock()
        app.root = Mock()
        app._return_home = SecurityQuizApp._return_home.__get__(app)

        app._return_home("終了")

        app.worker.set_tracked_boxes.assert_called_once_with(None)
        self.assertEqual(app.mode, "notice")

    def test_win_screen_remains_for_ten_seconds_then_returns_home(self):
        app = self.make_app("explanation", 10)
        app.question_index = 4
        app._continue_or_finish = SecurityQuizApp._continue_or_finish.__get__(app)

        with patch("app.time.monotonic", return_value=20):
            app._continue_or_finish()
        self.assertEqual(app.mode, "win")
        self.assertEqual(app.end_message, "おめでとう！全問正解！")
        with patch("app.time.monotonic", return_value=29.9):
            app._advance_phase()
        self.assertEqual(app.mode, "win")
        with patch("app.time.monotonic", return_value=30):
            app._advance_phase()
        self.assertEqual(app.mode, "home")

    def test_answer_shows_large_countdown(self):
        app = self.make_app("answer", 10)
        app.sets[1][0].update({"level": 1, "question": "問題文"})
        app.canvas = Mock()
        app.canvas.bbox.return_value = None
        app._render_faces = Mock()
        with patch("app.time.monotonic", return_value=11):
            app._render_game(1280, 760)

        timer = [
            call.kwargs for call in app.canvas.create_text.call_args_list
            if call.kwargs.get("text") == "4"
        ]
        self.assertEqual(len(timer), 1)
        self.assertEqual(timer[0]["font"][1], 256)
        self.assertEqual(timer[0]["fill"], "#FBBF24")
        panels = [call.kwargs for call in app.canvas.create_rectangle.call_args_list]
        self.assertEqual(
            [(panel.get("fill"), panel.get("outline")) for panel in panels[:3]],
            [("#0B1D26", "#00E5FF"), ("", "#059669"), ("", "#E11D48")],
        )
        header_coords = app.canvas.coords.call_args.args[1:]
        self.assertEqual(header_coords[0], app.canvas.create_rectangle.call_args_list[1].args[0])
        self.assertEqual(header_coords[2], app.canvas.create_rectangle.call_args_list[2].args[2])
        for symbol, x, color in (("〇", 320, "#059669"), ("×", 960, "#E11D48")):
            call = next(
                call for call in app.canvas.create_text.call_args_list
                if call.kwargs.get("text") == symbol
            )
            self.assertEqual(call.args[0], x)
            self.assertEqual(call.kwargs["fill"], color)
        timer_call = next(
            call for call in app.canvas.create_text.call_args_list
            if call.kwargs.get("text") == "4"
        )
        self.assertEqual(timer_call.args[:2], (640, 380))

    def test_selection_timer_continues_with_extra_face(self):
        for mode in ("answer", "reveal"):
            with self.subTest(mode=mode):
                app = self.make_app(mode, 10)
                app.sets[1][0].update({"level": 1, "question": "問題文"})
                app.canvas = Mock()
                app.canvas.bbox.return_value = None
                app._render_faces = Mock()
                app.extra_face_since = 11
                with patch("app.time.monotonic", return_value=14):
                    app._render_game(1280, 760)
                labels = [
                    call.kwargs.get("text")
                    for call in app.canvas.create_text.call_args_list
                ]
                if mode == "answer":
                    self.assertIn("1", labels)
                else:
                    self.assertFalse(any(text in {"0", "1", "2", "3", "4", "5"} for text in labels))
                bar = app.canvas.create_rectangle.call_args_list[-1]
                self.assertAlmostEqual(
                    bar.args[2], 40 + (1280 - 80) * 0.2
                )

    def test_reveal_uses_selection_layout_and_shows_correct_answer(self):
        app = self.make_app("reveal", 10)
        app.sets[1][0].update({"level": 1, "question": "問題文"})
        app.canvas = Mock()
        app.canvas.bbox.return_value = None
        app._render_faces = Mock()
        with patch("app.time.monotonic", return_value=11):
            app._render_game(1280, 760)
        labels = [call.kwargs.get("text") for call in app.canvas.create_text.call_args_list]
        self.assertIn("正解は「〇」　解説まで", labels)
        self.assertFalse(any(text in {"0", "1", "2", "3", "4", "5"} for text in labels))
        self.assertIn("〇", labels)
        self.assertIn("×", labels)
        panels = [call.kwargs for call in app.canvas.create_rectangle.call_args_list]
        self.assertEqual(
            [(panel.get("fill"), panel.get("outline")) for panel in panels[1:3]],
            [("#A7F3D0", "#059669"), ("", "#E11D48")],
        )
        bar = app.canvas.create_rectangle.call_args_list[-1]
        self.assertAlmostEqual(bar.args[2], 40 + (1280 - 80) * 0.8)
        app._render_faces.assert_called_once()

    def test_only_the_correct_reveal_zone_flashes(self):
        for answer, highlighted in (("〇", ["#A7F3D0", ""]), ("×", ["", "#FECDD3"])):
            for mode in ("answer", "reveal"):
                for elapsed in (0, 0.5, 1):
                    with self.subTest(answer=answer, mode=mode, elapsed=elapsed):
                        app = self.make_app(mode, 10)
                        app.sets[1][0].update({"level": 1, "answer": answer})
                        app.canvas = Mock()
                        app.canvas.bbox.return_value = None
                        app._render_faces = Mock()
                        with patch("app.time.monotonic", return_value=10 + elapsed):
                            app._render_game(1280, 760)
                        panels = app.canvas.create_rectangle.call_args_list[1:3]
                        fills = [panel.kwargs["fill"] for panel in panels]
                        expected = highlighted if mode == "reveal" and elapsed != 0.5 else ["", ""]
                        self.assertEqual(fills, expected)

    def test_reveal_flash_continues_with_extra_face(self):
        app = self.make_app("reveal", 10)
        app.sets[1][0]["level"] = 1
        app.extra_face_since = 10.5
        app.canvas = Mock()
        app.canvas.bbox.return_value = None
        app._render_faces = Mock()
        with patch("app.time.monotonic", return_value=12):
            app._render_game(1280, 760)
        self.assertEqual(app.canvas.create_rectangle.call_args_list[1].kwargs["fill"], "#A7F3D0")
        bar = app.canvas.create_rectangle.call_args_list[-1]
        self.assertAlmostEqual(bar.args[2], 40 + (1280 - 80) * 0.6)

    def test_reveal_status_does_not_show_countdown(self):
        app = self.make_app("reveal", 10)
        app.camera_error = None
        app.worker = Mock()
        app.worker.snapshot.return_value = (object(), [], 1, True, None)
        with patch("app.time.monotonic", return_value=11):
            first = app._status_text()
        with patch("app.time.monotonic", return_value=12):
            second = app._status_text()
        self.assertEqual(first, second)
        self.assertNotIn("残り", first)

    def test_reading_displays_large_question_or_explanation(self):
        app = self.make_app("question", 10)
        app.canvas = Mock()
        app.canvas.bbox.return_value = (100, 100, 800, 250)

        app._render_reading(960, 600, "問題", "問題文")

        text_calls = [call.kwargs for call in app.canvas.create_text.call_args_list]
        self.assertEqual(text_calls[0]["text"], "問題")
        self.assertEqual(text_calls[1]["text"], "問題文")
        self.assertGreaterEqual(text_calls[1]["font"][1], 28)
        self.assertEqual(text_calls[1]["fill"], "#F0F9FF")
        self.assertEqual(app.canvas.create_rectangle.call_count, 1)

        app.canvas.create_text.reset_mock()
        app._render_reading(960, 600, "解説", "解説文")
        text_calls = [call.kwargs for call in app.canvas.create_text.call_args_list]
        self.assertIn("自動", text_calls[1]["text"])
        self.assertEqual(text_calls[2]["text"], "解説文")
        self.assertEqual(text_calls[2]["fill"], "#BAE6FD")
        app.canvas.create_text.reset_mock()
        app.correct_ids = set()
        app.wrong_ids = set()
        app._render_reading(960, 600, "解説", "解説文")
        guidance = app.canvas.create_text.call_args_list[1].kwargs["text"]
        self.assertIn("F8", guidance)
        self.assertIn("F9", guidance)

    def test_error_screen_is_nonmodal_and_retains_reason(self):
        app = self.make_app("answer", 10)
        app.root = Mock()
        app._return_home = SecurityQuizApp._return_home.__get__(app)

        with patch("app.time.monotonic", return_value=20):
            app._return_home("カメラエラー")
        self.assertEqual(app.mode, "notice")
        self.assertEqual(app.end_message, "カメラエラー")
        app.root.after_idle.assert_not_called()
        with patch("app.time.monotonic", return_value=25):
            app._advance_phase()
        self.assertEqual(app.mode, "home")
        self.assertEqual(app.home_message, "カメラエラー")

    def test_countdown_expiry_without_faces_returns_home_and_keeps_same_set(self):
        app = self.make_countdown_app([])

        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()

        self.assertEqual(app.mode, "home")
        self.assertEqual(app.next_set_id, 3)
        app._begin_question.assert_not_called()


class CameraSwitchTests(unittest.TestCase):
    def test_switch_releases_old_camera_and_starts_selected_camera(self):
        app = SecurityQuizApp.__new__(SecurityQuizApp)
        first = CameraDevice(0, "Built-in Camera")
        second = CameraDevice(1, "USB Camera")
        previous_worker = Mock()
        previous_worker.stop.return_value = True
        app.mode = "home"
        app.camera_devices = [first, second]
        app.camera_index = first.index
        app.worker = previous_worker
        app.camera_var = Mock()
        app.camera_error = None
        app.latest_frame = object()
        app.latest_faces = [(1, 2, 3, 4)]
        app.latest_detections = [Detection(0.5, 0.5, (0.4, 0.4, 0.2, 0.2))]
        app.last_sequence = 12
        app._render = Mock()
        replacement = Mock()

        with patch("app.CameraWorker", return_value=replacement) as worker_type:
            app._switch_camera(second)

        previous_worker.stop.assert_called_once_with()
        worker_type.assert_called_once_with(second.index)
        replacement.start.assert_called_once_with()
        self.assertIs(app.worker, replacement)
        self.assertEqual(app.camera_index, second.index)
        self.assertIsNone(app.latest_frame)
        self.assertEqual(app.latest_faces, [])
        self.assertEqual(app.latest_detections, [])
        self.assertEqual(app.last_sequence, -1)
        app.camera_var.set.assert_called_once_with(second.label)


if __name__ == "__main__":
    unittest.main()
