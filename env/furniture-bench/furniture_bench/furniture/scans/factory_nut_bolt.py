from pathlib import Path
from glob import glob
import numpy as np
import json

from furniture_bench.utils.pose import get_mat
from furniture_bench.config import config
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.scans.parts.factory_nut import FactoryNut
from furniture_bench.furniture.scans.parts.factory_bolt import FactoryBolt

SCAN_ASSET_ROOT = Path(__file__).parent.parent.parent.absolute() / "assets_no_tags"


class FactoryNutBolt(Furniture):
    def __init__(self):
        super().__init__()
        furniture_conf = config["furniture"]["factory_nut_bolt"]
        self.furniture_conf = furniture_conf

        self.tag_size = furniture_conf["tag_size"]

        self.parts = [
            FactoryNut(furniture_conf["factory_nut"], 0),
            FactoryBolt(furniture_conf["factory_bolt"], 1),
        ]
        self.num_parts = len(self.parts)

        self.should_be_assembled = [(0, 1)]
        self.skill_attach_part_idx = 0

        assembly_json_pattern = str(
            SCAN_ASSET_ROOT / furniture_conf["assembly_json_fname"]
        )
        # Get assembly files that might be there in addition
        assembly_json_fnames = glob(
            assembly_json_pattern.replace(".json", "") + "**.json"
        )

        rel_pose_mat_list = []

        if len(assembly_json_fnames):
            for i, assembly_json_fname in enumerate(assembly_json_fnames):
                with open(assembly_json_fname, "r") as f:
                    assembly_data = json.load(f)["data"]

                ref_pose_mat = np.asarray(assembly_data["reference"]["pose"]).reshape(
                    4, 4
                )
                moved_pose_mat = np.asarray(assembly_data["moved"]["pose"]).reshape(
                    4, 4
                )

                rel_pose_mat = np.linalg.inv(ref_pose_mat) @ moved_pose_mat
                rel_pose_mat_list.append(rel_pose_mat)

            self.assembled_rel_poses[(0, 1)] = np.asarray(rel_pose_mat_list).reshape(
                -1, 4, 4
            )
        else:
            print(
                f"Warning! FactoryNutBolt furniture class, no file names matching pattern {assembly_json_pattern} for obtaining assembly rewards"
            )
            self.assembled_rel_poses[(0, 1)] = np.eye(4).reshape(1, 4, 4)
