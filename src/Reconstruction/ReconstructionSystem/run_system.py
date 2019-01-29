# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/ReconstructionSystem/run_system.py

import os
import sys
import json
import argparse
import time, datetime

from src.paths import utility
print(utility)
sys.path.append(utility)
from file import *
from initialize_config import *

def run_system(config, make, register, refine, integrate, debug_mode):

    if not make and \
            not register and \
            not refine and \
            not integrate:
        parser.print_help(sys.stderr)
        sys.exit(1)


    if config is not None:
        with open(config) as json_file:
            config = json.load(json_file)
            initialize_config(config)
            check_folder_structure(config["path_dataset"])
    assert config is not None

    if debug_mode:
        config['debug_mode'] = True
    else:
        config['debug_mode'] = False

    print("====================================")
    print("Configuration")
    print("====================================")
    for key, val in config.items():
        print("%40s : %s" % (key, str(val)))

    times = [0,0,0,0]
    if make:
        start_time = time.time()
        import make_fragments
        make_fragments.run(config)
        times[0] = time.time() - start_time
    if register:
        start_time = time.time()
        import register_fragments
        register_fragments.run(config)
        times[1] = time.time() - start_time
    if refine:
        start_time = time.time()
        import refine_registration
        refine_registration.run(config)
        times[2] = time.time() - start_time
    if integrate:
        start_time = time.time()
        import integrate_scene
        mesh = integrate_scene.run(config)
        times[3] = time.time() - start_time

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    sys.stdout.flush()
    return mesh
