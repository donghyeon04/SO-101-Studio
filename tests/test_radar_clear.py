"""결정 0039: radar_clear 대기 자세의 조건 (링크 점 작업대 y 120mm 이하, 손끝 150mm 이상, 관절 한계와 안전 검사 안)."""
import pytest

from so101_studio.arm.sim_driver import SimDriver
from so101_studio.core.config import load_robot, load_workspace
from so101_studio.core.types import JointState
from so101_studio.service.arm_service import ArmService

CFG = load_robot("config/robot.yaml")
WS = load_workspace("tests/workspace_600x400.yaml")


@pytest.fixture
def svc(tmp_path):
    s = ArmService(SimDriver(CFG.poses["home"]), CFG, WS, speed_pct=100, log_root=str(tmp_path))
    assert s.connect()[0]
    yield s
    s.disconnect()


def test_radar_clear_pose_meets_decision_0039(svc):
    q = JointState(*CFG.poses["radar_clear"])
    assert svc.ctl.safety.check_goal(q) is None  # 관절 한계·작업 영역·바닥 안
    assert all(y <= 120 for _, y in svc.arm_links(q))  # 어깨·팔꿈치·손목·손끝
    assert svc.tool_position(q)[2] >= 150


def test_radar_clear_reachable_from_home_and_back(svc):
    assert svc.skills.radar_clear().status == "ok"
    assert svc.driver.read_joints().elbow_flex == pytest.approx(CFG.poses["radar_clear"][2], abs=2.0)
    assert svc.park().status == "ok"


def test_radar_clear_from_a_low_reach_on_the_right_edge(tmp_path):
    """REQ-117: checktest a1 30x100 S3 확인 뒤 자세(손끝 약 (386, 94), 손목 93.9°)에서 한꺼번에 가면 손목이 뒤집히며 x 경계를 넘는다.
    팔 올리기 → 어깨 회전 → 손목 순서로 가서 radar_clear에 닿아야 한다."""
    s = ArmService(SimDriver([66.94, -3.55, -0.33, 93.88, 12.91, 3.0]), CFG, load_workspace(), speed_pct=100, log_root=str(tmp_path))
    assert s.connect()[0]
    try:
        assert s.skills.radar_clear().status == "ok"
        assert s.driver.read_joints().wrist_flex == pytest.approx(CFG.poses["radar_clear"][3], abs=2.0)
    finally:
        s.disconnect()
