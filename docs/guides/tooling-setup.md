# 선택 설치

상태: 확정

## C24. MuJoCo (물리 시뮬레이션)

| # | 이름 | 종류 | 용도 | 설치 |
|---|---|---|---|---|
| C24 | **MuJoCo** | 파이썬 패키지(MuJoCo 세션 PC) | 물리 시뮬레이션(결정 0053, `mujoco-spec.md` 13.8) | so101 환경에서 `pip install --no-deps "mujoco>=4.0,<5" mujoco-core mujoco-studio glfw pyopengl`(numpy가 바뀌지 않게). 버전 고정(M2, `pip freeze` 2026-11-13): `pip install --no-deps mujoco==4.0.0 mujoco-core==4.0.0 mujoco-studio==4.0.0 glfw==2.11.0 pyopengl==3.1.10`. 설치 뒤 `python -c "import numpy; print(numpy.__version__)"`가 2.2.6 |

### C-8. 학습 플랫폼 
| # | 이름 | 어디에 | 설치 |
|---|---|---|---|
| C25 | `lerobot[dataset]` | 앱 환경 `so101` | 먼저 `pip install --dry-run "lerobot[dataset]==0.6.1"`로 numpy가 바뀌지 않는지 본다. 바뀌면 그 목록에서 빠진 패키지만 `pip install --no-deps`로 넣는다. 설치 뒤 `python -c "import numpy; print(numpy.__version__)"`가 2.2.6이고 `from lerobot.datasets.lerobot_dataset import LeRobotDataset`이 된다(모듈 경로는 설치본에서 확인). 넣은 버전을 `environment.yml`에 `==`로 적는다(팔 세션 PR) |
| C26 | 학습 환경 `so101-train` | 새 conda 환경(Python 3.12) | CUDA torch(RTX 5070 Ti는 CUDA 12.8 이상 빌드, PyTorch 공식 설치 명령의 해당 인덱스), `lerobot==0.6.1`(학습에 필요한 extra), `mujoco==4.0.0`. `python -c "import torch; print(torch.cuda.is_available())"`가 True. 버전을 새 `environment-train.yml`에 `==`로 고정한다. 앱 환경 `so101`의 torch·numpy는 건드리지 않는다 |
| C27 | `so101-nexus-mujoco==0.3.12` | `so101-train` | 강화학습 데모용. `teleop` extra는 쓰지 않는다(`lerobot<0.6`을 요구한다). 베타라 버전을 고정한다. Apache-2.0 |

