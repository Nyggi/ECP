from Config import Config
from DataHandler import DataHandler

cfg = Config()
cfg.WEKA_FEATURES = False
cfg.WRITE_CSV = True

for i in range(24):
    print(i)
    cfg.HOUR_TO_PREDICT = i
    dh = DataHandler(cfg)