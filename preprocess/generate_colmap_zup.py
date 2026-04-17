# 功能：兼容旧入口，转发到独立的 Z-up 空三模块目录，方便后续继续扩展空三相关脚本。

import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocess.zup_pipeline.colmap_zup_runner import main


if __name__ == "__main__":
    main()
