# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

RESULT_FILENAMES = [
    # "mesh_clearGrasp-test.csv"
    # "ngp_clearGrasp-test.csv"
    # "foundationposepartial_CNCpicking-test.csv"
    "Megapose_CNCpicking-test.csv"
]

# Folder with the BOP datasets.
if "BOP_PATH" in os.environ:
    datasets_path = os.environ["BOP_PATH"]
else:
    datasets_path = r"/home/zemanvit/Projects/megapose6d"

# Folder with pose results to be evaluated.
# results_path = r"/path/to/folder/with/results"
# results_path = r"/home/testbed/Projects/bop_toolkit/clearGrasp/results_csv"
results_path = r"/home/zemanvit/Projects/megapose6d/CNCpicking/results"

# Folder for the calculated pose errors and performance scores.
# eval_path = r"/path/to/eval/folder"
# eval_path = r"/home/testbed/Projects/bop_toolkit/clearGrasp/results_metrics"
eval_path = r"/home/zemanvit/Projects/megapose6d/CNCpicking/results_metrics"

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r"/home/testbed/Projects/bop_toolkit"

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r"/path/to/bop_renderer/build"

# Executable of the MeshLab server.
meshlab_server_path = r"/path/to/meshlabserver.exe"
