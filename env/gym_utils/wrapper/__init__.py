from .multi_step import MultiStep
from .multi_step_full import MultiStepFull
from .robomimic_lowdim import RobomimicLowdimWrapper
from .robomimic_image import RobomimicImageWrapper
from .d3il_lowdim import D3ilLowdimWrapper
from .mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper
# from .pusht_state import PushTStateWrapper
# from .pusht_image import PushTImageWrapper


wrapper_dict = {
    "multi_step": MultiStep,
    "multi_step_full": MultiStepFull,
    "robomimic_lowdim": RobomimicLowdimWrapper,
    "robomimic_image": RobomimicImageWrapper,
    "d3il_lowdim": D3ilLowdimWrapper,
    "mujoco_locomotion_lowdim": MujocoLocomotionLowdimWrapper,
    # "pusht_state": PushTStateWrapper,
    # "pusht_image": PushTImageWrapper,
}
