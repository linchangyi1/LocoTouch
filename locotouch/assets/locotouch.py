from .go1 import Go1_CFG

# locotouch
LocoTouch_CFG = Go1_CFG.copy()
LocoTouch_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch.usd"

LocoTouch_Instanceable_CFG = Go1_CFG.copy()
LocoTouch_Instanceable_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_instanceable.usd"


# locotouch without tactile sensors
LocoTouch_Without_Tactile_CFG = Go1_CFG.copy()
LocoTouch_Without_Tactile_CFG.spawn.usd_path = "locotouch/assets/locotouch/locotouch_without_tactile.usd"

LocoTouch_Without_Tactile_Instanceable_CFG = Go1_CFG.copy()
LocoTouch_Without_Tactile_Instanceable_CFG.spawn.usd_path="locotouch/assets/locotouch/locotouch_without_tactile_instanceable.usd"

