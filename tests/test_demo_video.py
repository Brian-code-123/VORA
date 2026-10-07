"""The demo video must show what the brief asks for, stay within two minutes and never dress up a result.
Only the pure parts of scripts/make_demo_video.py are tested here; recording needs Chrome, ffmpeg and the models."""
import json
from pathlib import Path

from scripts import make_demo_video as V

ROOT = Path(__file__).resolve().parent.parent


def all_text() -> str:
    return " ".join(s.caption for sc in V.SCENES for s in sc.steps).lower()


def test_scenes_cover_every_item_the_brief_asks_the_video_to_show():
    """Brief, deliverable 3: user speaks into a microphone; ASR shows partial text in real time; RAG retrieves context and
    generates a response; TTS plays audio; latency is highlighted ("Response time")."""
    text = all_text()
    for needle in ("microphone", "partial", "retriev", "knowledge base", "token", "tts", "response time", "1.5"):
        assert needle in text, f"no caption mentions: {needle}"


def test_scenes_also_show_streaming_extras_the_brief_lists():
    text = all_text()
    assert "barge-in" in text and "not sure" in text and "mandarin" in text


def test_steps_use_known_triggers_and_are_numbered_in_order():
    for sc in V.SCENES:
        assert sc.steps and all(s.trigger in V.TRIGGERS for s in sc.steps), sc.id
    english = next(sc for sc in V.SCENES if sc.id == "english")
    assert [s.trigger for s in english.steps] == ["start", "partial", "context", "tokens", "speaking", "latency"]


def test_total_length_budget_is_within_two_minutes():
    assert 90 <= V.total_budget_s() <= 120, V.total_budget_s()


def test_frame_durations_hold_the_last_frame_and_never_go_below_one_tick():
    assert V.frame_durations([0.0, 0.5, 1.0], end=2.0) == [0.5, 0.5, 1.0]
    d = V.frame_durations([0.0, 0.0, 0.001, 1.0], end=1.5)
    assert all(x >= 1 / 60 for x in d) and len(d) == 4


def test_concat_list_repeats_the_last_file_as_ffmpeg_requires():
    txt = V.concat_list(["a.jpg", "b.jpg"], [0.5, 1.0])
    assert txt.splitlines() == ["file 'a.jpg'", "duration 0.5", "file 'b.jpg'", "duration 1.0", "file 'b.jpg'"]


def test_mix_filter_places_both_tracks_by_their_offsets_and_clamps_negatives():
    f = V.mix_filter(user_offset_ms=1500, tts_offset_ms=-40)
    assert "adelay=1500:all=1" in f and "adelay=0:all=1" in f and "amix=inputs=2:normalize=0" in f


def test_pick_take_uses_the_median_latency_and_ignores_failed_takes():
    takes = [{"ok": True, "latency_ms": 900}, {"ok": True, "latency_ms": 2900}, {"ok": True, "latency_ms": 1100}, {"ok": False, "latency_ms": 10}]
    assert V.pick_take(takes)["latency_ms"] == 1100
    assert V.pick_take([{"ok": True, "latency_ms": 5}])["latency_ms"] == 5


def test_pick_take_with_no_successful_take_is_an_error():
    import pytest
    with pytest.raises(ValueError):
        V.pick_take([{"ok": False, "latency_ms": 1}])


def test_gate_rows_keep_every_status_and_number_from_gates_final_json():
    gates = json.loads((ROOT / "results" / "gates_final.json").read_text())
    rows = V.gate_rows(gates)
    assert [r[0] for r in rows] == [g["id"] for g in gates]
    word = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}
    for (gid, _name, status, measured), g in zip(rows, gates):
        assert status == word[g["ok"]] and measured == g["measured"], gid


def test_script_markdown_lists_every_caption_and_says_the_microphone_is_simulated():
    md = V.script_markdown(V.SCENES, V.CARDS)
    for sc in V.SCENES:
        for s in sc.steps:
            assert s.caption.split("{")[0].strip() in md
    assert "fake microphone" in md.lower() and "120" in md


def test_a_real_human_voice_scene_is_included_and_honest_about_phone_speech():
    sc = next(s for s in V.SCENES if s.id == "real_voice")
    text = " ".join(s.caption for s in sc.steps).lower()
    assert "real human" in text and "minds-14" in text
    assert "{outcome}" in text, "the caption must say what VORA actually did (refused or answered), not what we hoped"
    assert all(Path(ROOT / u.text).exists() for u in sc.mic if u.voice == "file")


def test_outcome_text_follows_whether_context_was_found():
    assert "not sure" in V.outcome_text([])
    assert "loosely related" in V.outcome_text(["E13"])


def test_limits_card_admits_that_off_topic_questions_are_not_always_refused():
    card = V.limits_card().lower()
    assert "off-topic" in card and "60%" in card


def test_barge_in_question_starts_while_the_first_answer_is_still_playing():
    """The first question ends about 2.6 s after it starts and its answer lasts well over 4 s: the second must start inside it."""
    sc = next(s for s in V.SCENES if s.id == "barge_in")
    first, second = sc.mic
    assert 5.5 <= second.at_s - first.at_s <= 6.5, second.at_s - first.at_s


def test_gates_card_does_not_truncate_the_measured_text():
    gates = json.loads((ROOT / "results" / "gates_final.json").read_text())
    card = V.gates_card(gates)
    assert next(g for g in gates if g["id"] == "G7")["measured"].split(";")[0] in card
    assert "pi_class_measured: unverified" in card


# ---- fidelity: the video must show what actually happened, in the order it happened -------------------------------
def test_latency_step_needs_a_new_metrics_message_not_a_different_number():
    """Two questions can have the same latency (both 0.68 s in a recording): comparing the shown text then reuses the first
    question's number for the second. A new `metrics` message is the only proof that the new question was answered."""
    assert V.latency_due(metrics=1, base=1) is False
    assert V.latency_due(metrics=2, base=1) is True
    assert V.latency_due(metrics=0, base=0) is False


def test_response_time_text_matches_the_page_format():
    assert V.response_time_text(680.0) == "0.68 s" and V.response_time_text(1215.0) == "1.22 s" and V.response_time_text(None) == "–"


def test_english_scene_shows_tts_playing_before_it_highlights_the_response_time():
    english = next(sc for sc in V.SCENES if sc.id == "english")
    assert [s.trigger for s in english.steps] == ["start", "partial", "context", "tokens", "speaking", "latency"]
    speaking = next(s for s in english.steps if s.trigger == "speaking")
    assert "synthesi" in speaking.caption.lower() and "play" in speaking.caption.lower()


def test_every_scene_that_answers_has_a_speaking_or_barge_in_step_before_its_latency_step():
    for sc in V.SCENES:
        trig = [s.trigger for s in sc.steps]
        assert trig.index("latency") > 0 and trig[-1] == "latency", sc.id
        if sc.id != "real_voice" and sc.id != "refusal":
            assert "speaking" in trig[:trig.index("latency")], sc.id


def test_barge_in_latency_caption_says_it_is_the_new_question_and_the_scene_waits_for_its_answer():
    sc = next(s for s in V.SCENES if s.id == "barge_in")
    assert sc.steps[-1].trigger == "latency" and "new question" in sc.steps[-1].caption.lower()
    assert V.TAIL_S >= 4.0, "the second answer must be heard before the scene ends"
    assert sc.budget_s >= 22


def test_mean_volume_of_a_window_tells_sound_from_silence(tmp_path):
    import shutil
    import subprocess
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    wav = tmp_path / "t.wav"      # 1 s of tone, then 1 s of silence
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-f", "lavfi", "-i",
                    "anullsrc=r=44100:cl=mono", "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]", "-map", "[a]", "-t", "2", str(wav)], check=True)
    assert V.window_mean_volume(wav, 0.1, 0.8) > -30
    assert V.window_mean_volume(wav, 1.2, 0.6) < -70


def test_video_check_flags_a_scene_with_no_sound_after_its_last_step():
    ok = V.check_scene_audio("english", mean_db=-25.0)
    bad = V.check_scene_audio("barge_in", mean_db=-91.0)
    assert ok is None and "barge_in" in bad and "no sound" in bad


def test_steps_to_fire_stops_after_barge_in_so_the_next_step_is_judged_on_fresh_state():
    """Bug seen in the video: `barge_in` and `latency` lit up in the same poll, and the latency step still used the FIRST
    question's metrics. After barge_in fires, the remaining steps must wait for the next poll (with the new baseline)."""
    sc = next(s for s in V.SCENES if s.id == "barge_in")
    everything_due = {t: True for t in V.TRIGGERS}
    idx = {s.trigger: i for i, s in enumerate(sc.steps)}
    assert V.steps_to_fire(sc.steps, set(), everything_due) == [idx["start"], idx["speaking"], idx["barge_in"]]
    assert V.steps_to_fire(sc.steps, {0, 1, 2}, everything_due) == [idx["latency"]]


def test_steps_to_fire_keeps_script_order_and_waits_for_the_first_step_that_is_not_due():
    english = next(s for s in V.SCENES if s.id == "english")
    due = {t: True for t in V.TRIGGERS}
    due["context"] = False
    assert V.steps_to_fire(english.steps, set(), due) == [0, 1]
    assert V.steps_to_fire(english.steps, {0, 1}, due) == []
    due["context"] = True
    assert V.steps_to_fire(english.steps, {0, 1}, due) == [2, 3, 4, 5]
    assert V.steps_to_fire(english.steps, set(range(6)), due) == []


def test_scenes_that_show_an_answer_only_count_a_take_that_really_found_context():
    """A take where the ASR misheard the question and VORA said 'not sure' must not be chosen for a scene that claims 'answered'."""
    expect = {sc.id: sc.expect_context for sc in V.SCENES}
    assert expect == {"english": True, "mandarin": True, "barge_in": True, "refusal": False, "real_voice": False}
    assert V.take_ok(fired=6, steps=6, ctx=["E03"], expect_context=True) is True
    assert V.take_ok(fired=6, steps=6, ctx=[], expect_context=True) is False
    assert V.take_ok(fired=4, steps=6, ctx=["E03"], expect_context=True) is False
    assert V.take_ok(fired=4, steps=4, ctx=[], expect_context=False) is True


def test_the_barge_in_question_is_a_plain_question_that_contains_a_word_the_tts_said_first():
    sc = next(s for s in V.SCENES if s.id == "barge_in")
    assert sc.mic[1].text == "What is the wake word?"      # works since the echo-guard fix; no lead-in needed


def test_latency_card_lists_every_take_and_marks_the_rejected_ones():
    """Hiding the slow rejected take (10 s in one recording) would flatter the numbers: show it, struck through, with the reason."""
    takes = {"english": {"latency_ms": 670.0, "n": 3, "all": [{"ok": True, "latency_ms": 670.0}, {"ok": False, "latency_ms": 10027.0}, {"ok": True, "latency_ms": 700.0}]}}
    gates = json.loads((ROOT / "results" / "gates_final.json").read_text())
    card = V.latency_card(takes, gates, None)
    assert "10.03" in card and "<s>10.03</s>" in card
    assert "2 of 3 takes usable" in card and "misheard" in card.lower()


def test_barge_in_caption_does_not_claim_a_server_cancel_that_may_not_have_happened():
    """If the first answer was already fully generated, nothing is left to cancel on the server: what always happens is that
    the page stops playing it the moment the new speech is recognised."""
    sc = next(s for s in V.SCENES if s.id == "barge_in")
    text = next(s.caption for s in sc.steps if s.trigger == "barge_in").lower()
    assert "playback stops" in text and "if they are still running" in text


def test_the_page_latches_the_barge_in_flag_so_a_slow_poll_cannot_miss_it():
    assert "D.bargeIn = true" in V.INIT_JS and "setInterval" in V.INIT_JS
    assert "bargeInSeen" in V.SNAPSHOT_JS or "__demo || {}).bargeIn" in V.SNAPSHOT_JS


def test_limits_card_reports_the_measured_share_of_understood_interruptions():
    takes = {"barge_in": {"latency_ms": 937.0, "n": 5, "all": [{"ok": o, "latency_ms": 1000.0} for o in (False, False, True, False, True)]}}
    card = V.limits_card(takes).lower()
    assert "2 of 5" in card and "interrupt" in card
    assert "2 of 5" not in V.limits_card().lower()            # without measurements the card does not invent a number


def test_a_take_whose_script_panel_overflows_is_rejected():
    """Seen in the video: with six steps the last caption (the Response time line) was cut off below the panel."""
    assert V.take_ok(fired=6, steps=6, ctx=["E03"], expect_context=True, fits=False) is False
    assert V.take_ok(fired=6, steps=6, ctx=["E03"], expect_context=True, fits=True) is True


def test_the_page_can_tell_whether_the_script_panel_fits():
    assert "__demoFits" in V.INIT_JS and "scrollHeight" in V.INIT_JS and "fits" in V.SNAPSHOT_JS


def test_script_panel_text_is_small_enough_for_six_two_line_steps():
    """Panel height is 720 - top - bottom; six steps of two lines each plus the header must fit."""
    import re
    top = int(re.search(r'left: "16px", top: "(\d+)px"', V.INIT_JS).group(1))
    bottom = int(re.search(r'zIndex: 2147483646, boxSizing: "border-box",\s*font: \'500 (\d+)px', V.INIT_JS).group(1))
    height = 720 - top - 70
    line = bottom * 1.3
    need = 28 + 6 * (2 * line + 10)
    assert need <= height, (need, height)


def test_script_file_points_each_brief_item_at_the_step_that_shows_it():
    md = V.script_markdown(V.SCENES, V.CARDS)
    english = next(sc for sc in V.SCENES if sc.id == "english")
    n = {s.trigger: i for i, s in enumerate(english.steps, start=1)}
    assert f"| User speaks into a microphone | English · step {n['start']} |" in md
    assert f"| TTS synthesizes and plays the audio | English · step {n['speaking']} |" in md
    assert f"English · step {n['latency']}" in md.split("Response time")[1].split("\n")[0] or f"English · step {n['latency']}" in md
