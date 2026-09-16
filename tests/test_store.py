from pathlib import Path

from lecture_copilot.store import Store


def test_lecture_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.sqlite3")
    course = store.create_course("Thermo", "Europe/London", {"week1_start": "2026-09-07"})
    lec = store.start_lecture(course["id"], "small", "cpu", "battery", 47)
    assert lec["status"] == "recording"

    sid = store.add_segment(lec["id"], 0.0, 2.5, "hello", 0.9, "small", 0.0, [])
    store.add_segment(lec["id"], 2.5, 6.0, "problem set four is due next thursday", 0.8, "small", 1.2, ["due", "next thursday"])
    fid = store.add_flag(lec["id"], 5.0, 0.0, 20.0)
    gid = store.add_gap(lec["id"], 6.0, "paused")
    store.close_gap(gid, 9.0)
    store.end_lecture(lec["id"], 0.31, 44)

    segs = store.segments(lec["id"])
    assert segs[0]["id"] == sid
    assert segs[1]["trigger_terms"] == ["due", "next thursday"]
    assert store.flags(lec["id"])[0]["id"] == fid
    assert store.gaps(lec["id"])[0]["t1"] == 9.0
    ended = store.get_lecture(lec["id"])
    assert ended["status"] == "ended" and ended["asr_rtf_p95"] == 0.31 and ended["battery_end"] == 44
    assert store.list_lectures()[0]["course_name"] == "Thermo"
    store.close()
