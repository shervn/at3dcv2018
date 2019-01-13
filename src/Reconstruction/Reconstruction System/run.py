import os
import sys
import json
import argparse
import time, datetime
sys.path.append(".")
from run_system import run_system


if __name__ == "__main__":
    config = "config/realsense.json"
    make = True
    register = True
    refine = True
    integrate = True
    debug_mode = True

    run_system(config, make, register, refine, integrate, debug_mode)
