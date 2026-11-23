"""REQ-120(arm-software-spec 11.2): 수직 집기 계획이 손–몸 자기 접촉(self:)이면 같은 파지의 다른 손목 방향으로 다시 계획한다. 자기 접촉 검사를 켠 채 본다."""
import math
from dataclasses import replace

import pytest

from so101_studio.arm.sim_driver import SimDriver
from so101_studio.core.config import load_robot, load_workspace
from so101_studio.core.types import DetectedObject, JointState, MotionResult, ToolPose
from so101_studio.service.arm_service import ArmService

CFG, WS = replace(load_robot("config/robot.yaml", gripper="parallel"), self_clear_mm=0.0), load_workspace("config/workspace.yaml")


def svc(tmp_path, cfg=CFG):
    s = ArmService(SimDriver(cfg.poses["home"]), cfg, WS, speed_pct=100, log_root=str(tmp_path))
    assert s.connect()[0]
    ev = []
    s.ctl.on_event = lambda kind, level, code, data: ev.append((code, data))
    return s, ev


def box(x, y, angle, w, length):
    return DetectedObject("t", x, y, angle, w, length, 1.0, "manual", 40.0)


@pytest.mark.slow
def test_blocked_pick_turns_the_wrist_180_and_picks(tmp_path):
    s, ev = svc(tmp_path)
    try:
        o = box(232.6, 119.9, 120.0, 30.0, 100.0)  # 반경 100mm: 고른 yaw 30°는 손이 어깨에 닿고 -150°는 풀린다(11-22 격자 탐색)
        s.driver.grip_block = s.skills.gripper_for_width(30.0)
        r = s.skills.pick(o)
        assert r.status == "ok", r
        (alt,) = [d for c, d in ev if c == "ARM-PICK-YAW-ALT"]
        assert alt["from_yaw_deg"] == 30.0 and alt["yaw_deg"] == -150.0 and alt["reason"].startswith("self:")
        assert s.skills.last_pick_info["yaw_alt"] is True and s.skills.grip["yaw"] == -150.0  # 놓기는 고른 yaw로 계산한다
    finally:
        s.disconnect()


def test_square_uses_the_other_side(tmp_path):
    s, _ev = svc(tmp_path)
    sk = s.skills
    try:
        o = box(301.8, 71.4, 0.0, 40.0, 40.0)  # 반경 100mm 정사각형: +180°는 손목 한계, 다른 변은 풀린다
        yaw = sk._pick_yaw(o)
        alts = sk._pick_yaw_alts(o, yaw)
        assert len(alts) == 3 and alts[0] == pytest.approx((yaw + 360.0) % 360.0 - 180.0)
        assert abs((alts[1] - yaw + 90.0) % 180.0 - 90.0) == pytest.approx(90.0)  # 다른 변은 턱 축이 90° 돈다
    finally:
        s.disconnect()


def test_all_alternatives_blocked_is_out_of_bounds_with_the_first_reason(tmp_path):
    s, ev = svc(tmp_path)
    try:
        r = s.skills.pick(box(215.2, 121.4, 45.0, 40.0, 40.0))  # 정면 반경 100mm, 45° 정사각형: 네 방향 모두 어깨에 닿는다
        assert r.status == "out_of_bounds" and r.reason.startswith("self:") and not [c for c, _ in ev if c == "ARM-PICK-YAW-ALT"]
        assert s.driver.read_joints() == JointState(*CFG.poses["home"])  # 움직이지 않았다
    finally:
        s.disconnect()


def test_default_off_does_not_precheck(tmp_path):
    s, _ev = svc(tmp_path, load_robot("config/robot.yaml", gripper="parallel"))  # 기본 self_clear_mm None = 끔 (REQ-119)
    try:
        sk = s.skills
        o = box(232.6, 119.9, 120.0, 30.0, 100.0)
        yaw = sk._pick_yaw(o)
        p = sk._plan_pick(o, yaw, (math.cos(math.radians(yaw)), math.sin(math.radians(yaw))), sk._tilt_ok(o, yaw))
        assert sk._self_block(p) is None
    finally:
        s.disconnect()


@pytest.mark.slow
def test_tilted_alternative_that_fails_retries_vertical(tmp_path, monkeypatch):
    s, ev = svc(tmp_path)
    sk = s.skills
    try:
        o = box(232.6, 119.9, 120.0, 30.0, 100.0)
        real_plan, calls = sk._plan_pick, []
        monkeypatch.setattr(sk, "_tilt_ok", lambda obj, y: True)  # 모든 yaw가 기울일 수 있다고 본다
        monkeypatch.setattr(sk, "_plan_pick", lambda obj, y, m, tilt: MotionResult("ik_fail", 0.0, ToolPose(0, 0, 0, 0)) if tilt else real_plan(obj, y, m, False))  # 기울인 계획은 늘 손목 한계
        monkeypatch.setattr(sk, "_self_block", lambda p: None if calls.append(p) or len(calls) > 1 else "self:gripper_link-shoulder_link")  # 첫 수직 계획만 막힘
        s.driver.grip_block = sk.gripper_for_width(30.0)
        r = sk.pick(o)
        assert r.status == "ok", r
        (alt,) = [d for c, d in ev if c == "ARM-PICK-YAW-ALT"]
        assert alt["yaw_deg"] == -150.0 and alt["tilted"] is False  # +180° 기울임이 안 돼 같은 yaw 수직으로
        assert sk.last_pick_tilted is False and sk.last_pick_info["yaw_alt"] is True
    finally:
        s.disconnect()


@pytest.mark.slow
def test_yaw_alt_is_false_when_not_used(tmp_path):
    s, _ev = svc(tmp_path, load_robot("config/robot.yaml", gripper="parallel"))  # 자기 접촉 끔: 고른 yaw 그대로 집는다
    try:
        s.driver.grip_block = s.skills.gripper_for_width(30.0)
        assert s.skills.pick(box(232.6, 119.9, 120.0, 30.0, 100.0)).status == "ok"
        assert s.skills.last_pick_info["yaw_alt"] is False
    finally:
        s.disconnect()
