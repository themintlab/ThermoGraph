import os
from zgraph import load as zgraph_load
from zgraph.core.base import ZGraphNode
from thermograph.config import LIBRARY_DIR, UNARY_TDB_PATH

def load_library(model_name: str) -> ZGraphNode:
    """
    Loads a thermograph reference model from the configured centralized library directory.
    Example: load_library('unary/GHSERCU')
    """
    rel_name = model_name if model_name.endswith(".zg") else f"{model_name}.zg"
    target_path = os.path.join(LIBRARY_DIR, rel_name)
    if not os.path.exists(target_path) and model_name.startswith("unary/"):
        phase_name = model_name.split("/", 1)[1].removesuffix(".zg").upper()
        from thermograph.io.tdb import parse_tdb
        nodes = parse_tdb(UNARY_TDB_PATH, phases=[phase_name])
        if phase_name in nodes:
            return nodes[phase_name]
    return zgraph_load(model_name, library_dir=LIBRARY_DIR)

