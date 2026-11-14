import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app import (
    CameraDevice,
    CameraWorker,
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
        profile.detectMultiScale.side_effect = [[], [(20, 10, 40, 40)]]

        boxes = detect_faces(image, frontal, profile, ((0.18, 0.4, 0.04, 0.133),))

        self.assertEqual(boxes, [(320, 70, 40, 40)])

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
            [(20, 10, 40, 40)],
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

    def test_extra_or_ambiguous_faces_fail_closed(self):
        tracker = FaceTracker([self.face(0.2), self.face(0.8)])
        with self.assertRaises(TrackingError):
            tracker.update([self.face(0.2), self.face(0.5), self.face(0.8)])

        tracker = FaceTracker([self.face(0.45), self.face(0.55)])
        with self.assertRaises(TrackingError):
            tracker.update([self.face(0.5), self.face(0.501)])

    def test_lost_tracks_can_be_removed_after_exit(self):
        tracker = FaceTracker([self.face(0.2), self.face(0.8)])
        tracker.update([self.face(0.2)])
        tracker.retain({1})
        self.assertEqual(set(tracker.tracks), {1})

    def test_long_lost_track_cannot_be_reassociated(self):
        with patch("quiz_core.time.monotonic", return_value=10):
            tracker = FaceTracker([self.face(0.2)])
        with patch("quiz_core.time.monotonic", return_value=13.1):
            tracker.update([])
        with patch("quiz_core.time.monotonic", return_value=23.1):
            with self.assertRaises(TrackingError):
                tracker.update([self.face(0.2)])

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
        app.end_message = ""
        app._return_home = Mock()
        app._continue_or_finish = Mock()
        return app

    def test_answer_timeout_shows_result(self):
        app = self.make_app("answer", 10)
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")
        self.assertEqual(app.correct_ids, {1})
        self.assertEqual(app.wrong_ids, {2})

    def test_temporary_extra_face_pauses_and_resumes_without_judging(self):
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
        self.assertEqual(app.mode, "answer")

        app.latest_detections = [FaceTrackerTests.face(0.2)]
        with patch("app.time.monotonic", return_value=12):
            app._process_new_frame()
            self.assertEqual(app.phase_started, 7)
            app._advance_phase()
        self.assertIsNone(app.extra_face_since)
        self.assertEqual(app.mode, "reveal")

    def test_question_intro_waits_two_seconds_before_speaking_for_each_question(self):
        app = self.make_app("answer", 10)
        app.sets[1].append(dict(app.sets[1][0]))
        for index in (0, 1):
            app.question_index = index
            app.speech.speak.reset_mock()
            with patch("app.time.monotonic", return_value=20):
                app._begin_question()
            self.assertEqual(app.mode, "question_intro")
            with patch("app.time.monotonic", return_value=21.9):
                app._advance_phase()
            app.speech.speak.assert_not_called()
            with patch("app.time.monotonic", return_value=22):
                app._advance_phase()
            self.assertEqual(app.mode, "question")
            app.speech.speak.assert_called_once_with("問題テキスト まるかばつか。")

    def test_intro_and_recovery_screens_are_rendered(self):
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

        app.canvas.create_text.reset_mock()
        app.mode = "answer"
        app.face_loss_since = 10
        with patch("app.time.monotonic", return_value=14):
            app._render()
        labels = [call.kwargs.get("text") for call in app.canvas.create_text.call_args_list]
        self.assertIn("参加者の顔を確認しています", labels)
        self.assertIn("6", labels)

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

    def test_missing_face_pauses_answer_and_recovers_before_ten_seconds(self):
        app = self.make_app("answer", 8)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = []
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        self.assertEqual(app.face_loss_since, 11)
        with patch("app.time.monotonic", return_value=20.9):
            app._advance_phase()
            self.assertEqual(app.mode, "answer")
            app.latest_detections = [FaceTrackerTests.face(0.2)]
            app._process_new_frame()
        self.assertIsNone(app.face_loss_since)
        self.assertAlmostEqual(app.phase_started, 17.9)
        with patch("app.time.monotonic", return_value=22.8):
            app._advance_phase()
        self.assertEqual(app.mode, "answer")
        with patch("app.time.monotonic", return_value=22.9):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")
        app._return_home.assert_not_called()

    def test_missing_face_times_out_even_if_face_returns_at_deadline(self):
        for process_frame in (False, True):
            with self.subTest(process_frame=process_frame):
                app = self.make_app("question", 8)
                with patch("app.time.monotonic", return_value=10):
                    app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
                app.latest_detections = []
                with patch("app.time.monotonic", return_value=11):
                    app._process_new_frame()
                app.latest_detections = [FaceTrackerTests.face(0.2)]
                with patch("app.time.monotonic", return_value=21):
                    if process_frame:
                        app._process_new_frame()
                    else:
                        app._advance_phase()
                app._return_home.assert_called_once()
                self.assertIn("再検出できません", app._return_home.call_args.args[0])

    def test_missing_participant_pauses_but_wrong_players_may_leave(self):
        app = self.make_app("answer", 8)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([
                FaceTrackerTests.face(0.2), FaceTrackerTests.face(0.8)
            ])
        app.latest_detections = [FaceTrackerTests.face(0.2)]
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        self.assertEqual(app.face_loss_since, 11)

        app.mode = "exit"
        app.face_loss_since = None
        with patch("app.time.monotonic", return_value=12):
            app._process_new_frame()
        self.assertIsNone(app.face_loss_since)

    def test_overlapping_extra_face_and_missing_face_pause_only_once(self):
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
        self.assertEqual(app.phase_started, 8)
        self.assertIsNone(app.face_loss_since)
        self.assertIsNone(app.extra_face_since)

    def test_unsafe_reassociation_still_stops_game(self):
        app = self.make_app("question", 8)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = []
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        app.latest_detections = [FaceTrackerTests.face(0.8)]
        with patch("app.time.monotonic", return_value=15):
            app._process_new_frame()
        app._return_home.assert_called_once()
        self.assertIn("追跡", app._return_home.call_args.args[0])

    def test_persistent_extra_face_stops_game_without_judging(self):
        app = self.make_app("answer", 6)
        with patch("app.time.monotonic", return_value=10):
            app.tracker = FaceTracker([FaceTrackerTests.face(0.2)])
        app.latest_detections = [
            FaceTrackerTests.face(0.2),
            FaceTrackerTests.face(0.8),
        ]
        with patch("app.time.monotonic", return_value=11):
            app._process_new_frame()
        with patch("app.time.monotonic", return_value=13):
            app._advance_phase()
        app._return_home.assert_called_once()
        self.assertIn("新しい顔", app._return_home.call_args.args[0])
        self.assertEqual(app.mode, "answer")

    def test_extra_face_timeout_does_not_accept_late_recovery(self):
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
        with patch("app.time.monotonic", return_value=13):
            app._process_new_frame()
        app._return_home.assert_called_once()

    def test_answer_waits_for_missing_face_then_uses_fresh_detection(self):
        app = self.make_app("answer", 10)
        app.tracker.detected_ids.return_value = {1}
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "answer")
        app.tracker.answer.assert_not_called()

        app.tracker.detected_ids.return_value = {1, 2}
        with patch("app.time.monotonic", return_value=16):
            app._advance_phase()
        self.assertEqual(app.mode, "reveal")

    def test_answer_retry_times_out_when_no_face_returns(self):
        app = self.make_app("answer", 10)
        app.tracker.detected_ids.return_value = set()
        with patch("app.time.monotonic", return_value=17):
            app._advance_phase()
        app._return_home.assert_called_once()
        self.assertIn("未回答", app._return_home.call_args.args[0])

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

    def test_explanation_advances_to_exit_for_wrong_players(self):
        app = self.make_app("explanation", 10)
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        self.assertEqual(app.mode, "exit")

    def test_explanation_continues_when_everyone_is_correct(self):
        app = self.make_app("explanation", 10)
        app.wrong_ids = set()
        with patch("app.time.monotonic", return_value=15):
            app._advance_phase()
        app._continue_or_finish.assert_called_once_with()

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

    def test_win_screen_remains_for_five_seconds_then_returns_home(self):
        app = self.make_app("explanation", 10)
        app.question_index = 4
        app._continue_or_finish = SecurityQuizApp._continue_or_finish.__get__(app)

        with patch("app.time.monotonic", return_value=20):
            app._continue_or_finish()
        self.assertEqual(app.mode, "win")
        self.assertEqual(app.end_message, "おめでとう！全問正解！")
        with patch("app.time.monotonic", return_value=24.9):
            app._advance_phase()
        self.assertEqual(app.mode, "win")
        with patch("app.time.monotonic", return_value=25):
            app._advance_phase()
        self.assertEqual(app.mode, "home")

    def test_all_wrong_after_exit_shows_retry_screen(self):
        app = self.make_app("exit", 10)
        app.correct_ids = set()

        with patch("app.time.monotonic", return_value=13):
            app._advance_phase()

        self.assertEqual(app.mode, "lose")
        self.assertEqual(app.end_message, "残念！もう一度チャレンジして")
        app._return_home.assert_not_called()

    def test_answer_shows_large_countdown(self):
        app = self.make_app("answer", 10)
        app.sets[1][0].update({"level": 1, "question": "問題文"})
        app.canvas = Mock()
        app._render_faces = Mock()
        with patch("app.time.monotonic", return_value=11):
            app._render_game(1280, 760)

        timer = [
            call.kwargs for call in app.canvas.create_text.call_args_list
            if call.kwargs.get("text") == "4"
        ]
        self.assertEqual(len(timer), 1)
        self.assertGreaterEqual(timer[0]["font"][1], 60)

    def test_reading_displays_large_question_or_explanation(self):
        app = self.make_app("question", 10)
        app.canvas = Mock()
        app.canvas.bbox.return_value = (100, 100, 800, 250)

        app._render_reading(960, 600, "問題", "問題文")

        text_calls = [call.kwargs for call in app.canvas.create_text.call_args_list]
        self.assertEqual(text_calls[0]["text"], "問題")
        self.assertEqual(text_calls[1]["text"], "問題文")
        self.assertGreaterEqual(text_calls[1]["font"][1], 28)

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
