#!/usr/bin/env python3
"""
Usage:
    python script/train_distill_flow.py --config-name=ft_distill_residual_flow_mlp_img --config-dir=cfg/robomimic/finetune/transport
"""

import hydra
from omegaconf import DictConfig
import logging
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # only if you’re sure this runs before any CUDA use

log = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path=None, config_name=None)
def main(cfg: DictConfig):
    """
    Main training function using simplified hydra instantiation.
    
    The pretrained flow matching policy is now loaded automatically
    within the DistillRLModel using the base_policy_path parameter.
    
    Args:
        cfg: OmegaConf configuration object
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log.info("Starting Distilled Flow RL Training")
    log.info(f"Working directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    
    # Create training agent using the same pattern as script/run.py
    # The agent will automatically instantiate the model, which will
    # automatically load the pretrained policy using the base_policy_path
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    
    # Start training
    try:
        agent.run()
        log.info("Training completed successfully")
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 