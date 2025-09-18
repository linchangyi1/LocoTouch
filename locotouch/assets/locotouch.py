from .go1 import Go1_CFG

# locotouch
LocoTouch_CFG = Go1_CFG.copy()
LocoTouch_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch.usd"
# LocoTouch_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_instanceable_new_over255.usd"

LocoTouch_Instanceable_CFG = Go1_CFG.copy()
LocoTouch_Instanceable_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_instanceable.usd"
# LocoTouch_Instanceable_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_instanceable_new_over255.usd"


# locotouch without tactile sensors
LocoTouch_Without_Tactile_CFG = Go1_CFG.copy()
LocoTouch_Without_Tactile_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_without_tactile.usd"
# LocoTouch_Without_Tactile_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_instanceable_new_over255.usd"

LocoTouch_Without_Tactile_Instanceable_CFG = Go1_CFG.copy()
LocoTouch_Without_Tactile_Instanceable_CFG.spawn.usd_path="locotouch/assets/locotouch/locotouch_without_tactile_instanceable.usd"
# LocoTouch_Without_Tactile_Instanceable_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_instanceable_new_over255.usd"


# python locotouch/scripts/train.py --task Isaac-Locomotion-LocoTouch-v1 --num_envs=40 --headless --logger=tensorboard

# python locotouch/scripts/play.py --task Isaac-Locomotion-LocoTouch-Play-v1 --num_envs=20 --headless

