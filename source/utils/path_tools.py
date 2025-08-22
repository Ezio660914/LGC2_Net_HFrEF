# -*- coding: utf-8 -*-
import inspect
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_project_dir(project_name="LGC2_Net_HFrEF", file=None):
    if file is None:
        frame_info = inspect.getframeinfo(inspect.currentframe().f_back)
        file = frame_info.filename
    file_path = Path(os.path.abspath(file))
    while file_path.name != project_name:
        file_path = file_path.parent
        if file_path == file_path.parent:
            break
    return file_path
