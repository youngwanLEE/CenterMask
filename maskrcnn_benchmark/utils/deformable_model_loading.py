import torch
import logging

def load_deformable_vovnet(cfg, f):
    state_dict = torch.load(f, map_location=torch.device("cpu"))
    import re
    logger = logging.getLogger(__name__)
    logger.info("Remapping conv weights for deformable conv weights")
    layer_keys = sorted(state_dict.keys())
    for idx, stage_with_dcn in enumerate(cfg.MODEL.VOVNET.STAGE_WITH_DCN, 2):
        if not stage_with_dcn:
            continue
        for old_key in layer_keys:
            if "layer" in old_key and f"OSA{idx}" in old_key and \
                    "conv" in old_key:
                for param in ["weight", "bias"]:
                    if old_key.find(param) is -1:
                        continue
                    new_key = old_key.replace(
                        "conv.{}".format(param), "conv.conv.{}".format(param)
                    )
                logger.info("old_key: {}, new_key: {}".format(
                    old_key, new_key
                ))
                state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]

    if "model" not in state_dict:
        state_dict = dict(model=state_dict)

    return state_dict