import json
import uuid

import numpy as np

def write_data(root, data_dict):
    scene_id = uuid.uuid4().hex
    path = root / "scenes" / (scene_id + ".npz")
    assert not path.exists()
    np.savez_compressed(path, **data_dict)
    return scene_id