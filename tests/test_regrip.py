"""REQ-118 쥠 다시 확인(arm-software-spec 6.6): 놓기 직전 턱이 비었으면 stopped(dropped), 작업 실행은 FAULT 대신 다음으로 간다."""
import pytest
from test_task import OK, ListSource, StubSkills, obj, run
from test_task import WS as WS_TASK
from test_task_check import StubSource, UsSkills, cand, runner
from test_workbench_bin import rig

from so101_studio.arm.skills import DROPPED_REASON
from so101_studio.core.config import load_robot
from so101_studio.core.events import EventBus
from so101_studio.core.types import DetectedObject, MotionResult, ToolPose
from so101_studio.task.runner import TaskRunner

CFG = load_robot("config/robot.yaml", gripper="single_jaw")
OBJ = DetectedObject("a", 215.0, 160.0, 0.0, 30.0, 100.0, 1.0, "manual", 40.0)  # config/manual_objects.yaml 첫 물체
DROPPED = MotionResult("stopped", 0.0, ToolPose(0, 0, 0, 0), reason=DROPPED_REASON)


def carry(lose: bool):
    """집은 뒤 옮기는 중 물체가 빠지면(lose) 막던 것이 없어져 턱이 예압 목표까지 닫힌다."""
    d, ctl, sk = rig(CFG)
    d.grip_block = sk.gripper_for_width(30.0)
    assert sk.pick(OBJ, sk.place_targets(zone="right")).status == "ok"
    if lose:
        d.grip_block = None
    ev = []
    ctl.on_event = lambda kind, level, code, data: ev.append((code, level, data))
    return d, ctl, sk, ev


@pytest.mark.slow  # 집고 놓기 한 번에 15초
def test_object_lost_in_carry_is_dropped_opens_and_lifts():
    d, ctl, sk, ev = carry(lose=True)
    try:
        n = len(d.log)
        r = sk.place("right")
        assert r.status == "stopped" and r.reason == DROPPED_REASON, r
        warn = [x for x in ev if x[0] == "ARM-PLACE-DROPPED"]
        assert len(warn) == 1 and warn[0][1] == "WARN" and set(warn[0][2]) == {"held_gap_mm", "gripper", "grip_hold"}
        held = [x[2] for x in ev if x[0] == "ARM-HELD"]
        assert held == [{"state": "released", "id": "a", "reason": "dropped", "at": None}]
        assert sk.grip is None and sk.held is None and sk.placed == [] and sk.last_place_xy is None
        assert d.log[-1].gripper > d.log[n].gripper + 1.0  # 정상 놓기와 같이 열었다
        top = sk.kin.jaw_tips_at(d.log[-1])
        low = sk.kin.jaw_tips_at(min(d.log[n:], key=lambda q: sk.kin.fk(q).z_mm))
        assert min(top[0][2], top[1][2]) > min(low[0][2], low[1][2]) + 20.0  # 연 뒤 올라갔다
    finally:
        ctl.close()


@pytest.mark.slow  # 집고 놓기 한 번에 15초
def test_object_still_held_places_ok():
    _d, ctl, sk, ev = carry(lose=False)
    try:
        assert sk.place("right").status == "ok"
        assert not [x for x in ev if x[0] == "ARM-PLACE-DROPPED"]
    finally:
        ctl.close()


def test_regrip_is_skipped_without_grip_memory_or_for_tilted_grips():
    _d, ctl, sk = rig(CFG)
    try:
        assert sk._dropped(None) is None  # 옛 경로(집기 기억 없음)
        sk.ctl.grip_hold = sk._q().gripper  # 턱이 목표까지 닫힌 상태: 수직이면 dropped
        assert sk._dropped({"approach": None}) is not None
        assert sk._dropped({"approach": {"beta": 20.0}}) is None  # 기울여 쥔 병렬은 가를 수 없어 보지 않는다(REQ-118 M5)
        sk.ctl.grip_hold_relaxed = True  # 그리퍼 부하 WARN이 목표를 지금 위치로 맞췄다: 쥔 채로도 차이가 0
        assert sk._dropped({"approach": None}) is None
        sk.ctl.hold_gripper(sk._q().gripper)  # 새로 쥐면 플래그가 풀린다
        assert not sk.ctl.grip_hold_relaxed and sk._dropped({"approach": None}) is not None
        sk.ctl.grip_hold = None
        assert sk._dropped({"approach": None}) is None  # 예압 목표가 없다(드라이런 accept_no_object)
    finally:
        sk.ctl.grip_hold = None
        ctl.close()


class DropSkills(StubSkills):
    def __init__(self, drops):
        super().__init__()
        self.drops = drops

    def place(self, zone):
        self.calls.append("place")
        if self.drops:
            self.drops -= 1
            return DROPPED
        return OK


def test_manual_drop_is_not_fault_and_skips_that_object():
    src = ListSource([obj(1, 300, 170), obj(2, 300, 200)])
    res, seen = run(TaskRunner(DropSkills(1), src, WS_TASK, EventBus()))
    assert res["end"] == "no_objects" and res["picked"] == ["obj-2"] and res["skipped"] == {"obj-1": "dropped"} and res["failed"] == ["obj-1"]
    assert ("PLACE", "obj-1", "dropped") in [(e.state, e.object_id, e.result) for e in seen]


def test_ultrasonic_drop_goes_to_radar_clear_and_rescans_without_note_failure():
    src, sk = StubSource([[cand(1)]], rescans=[[cand(2)], []]), type("S", (UsSkills, DropSkills), {})(1)
    res, _ = run(runner(src, skills=sk))
    assert res["picked"] == ["us-2"] and src.failures == [] and res["failed"] == ["us-1"]
    i = src.calls.index("prefetch")
    assert src.calls[i + 1: i + 3] == ["scan", "rescan"]  # 떨어뜨린 뒤: 도는 prefetch를 scan으로 끝까지 받고 버린 뒤 rescan
    assert sk.calls.count("radar_clear") >= 2


def test_three_drops_end_with_repeated_drop():
    src = ListSource([obj(i, 300, 160 + 10 * i) for i in range(1, 5)])
    res, seen = run(TaskRunner(DropSkills(3), src, WS_TASK, EventBus()))
    assert res["end"] == "repeated_drop" and seen[-1].result == "repeated_drop"



def test_overload_warn_marks_the_hold_relaxed():
    """부하 WARN(결정 0033 첫 단계)이 목표를 지금 위치로 맞추면 플래그가 서고, 그 뒤 놓기의 다시 확인은 건너뛴다."""
    _d, ctl, sk = rig(CFG)
    try:
        ctl.hold_gripper(sk._q().gripper - 2.0)
        ctl.grip_warn_pct, ctl.grip_load_s = 50.0, 0.0
        ctl._check_grip({"load_pct": {"gripper": 90.0}}, 0.0)
        assert ctl.grip_hold_relaxed and ctl.grip_hold == sk._q().gripper
        assert sk._dropped({"approach": None}) is None
    finally:
        ctl.grip_warn_pct, ctl.grip_hold = None, None
        ctl.close()


def test_stop_during_a_dropped_place_ends_stopped_not_fault():
    sk = DropSkills(1)
    tr = TaskRunner(sk, ListSource([obj(1, 300, 170)]), WS_TASK, EventBus())
    real = sk.place

    def place(zone):
        tr._stop = True  # 놓는 동안 정지를 눌렀다
        return real(zone)

    sk.place = place
    res, seen = run(tr)
    assert res["end"] == "STOPPED" and "FAULT" not in [e.state for e in seen]


@pytest.mark.slow
def test_sim_ultrasonic_drop_rescans_and_still_picks_all(tmp_path):
    """실제 UltrasonicSource(가상 ESP32): 첫 놓기에서 떨어뜨리면(물체는 가상 작업대에 남는다) prefetch를 받고 rescan해 끝내 셋을 다 집는다."""
    from test_task_check import sim_studio

    st, src, arm = sim_studio(tmp_path)
    real, n = arm.skills.place, []

    def place(zone):
        n.append(1)
        if len(n) == 1:
            arm.skills._release("dropped", None)
            return DROPPED
        return real(zone)

    arm.skills.place = place
    try:
        seen = []
        st.bus.subscribe("task", seen.append)
        res = st.start(blocking=True)
        assert len(res["picked"]) == 3 and res["failed"] == [res["failed"][0]], res
        states = [(e.state, e.result) for e in seen]
        assert ("PLACE", "dropped") in states and ("FAULT", None) not in states and not src.esp.boxes
    finally:
        arm.disconnect()
