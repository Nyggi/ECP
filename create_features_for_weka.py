from Config import Config
from DataHandler import DataHandler

def write_to_csv():
    with open(filepath, 'w') as f:
        f.write(header)

        for i in range(len(content_for_csv)):
            f.write(content_for_csv[i])

cfg = Config()
cfg.WEKA_FEATURES = False

if len(cfg.WEKA_HOUSEHOLD_IDS) > 1:
    amount_of_households = 'multiple'
else:
    amount_of_households = 'single'

for i in range(24):
    print(i)

    filepath = 'WEKA_features/features_for_WEKA_' + str(amount_of_households) + '/PandasData' + str(i) + '.csv'
    content_for_csv = []
    header = ''

    for household in cfg.WEKA_HOUSEHOLD_IDS:
        cfg.HOUR_TO_PREDICT = i
        cfg.HOUSE_ID = household

        dh = DataHandler(cfg)

        if header == '':
            header = dh._get_csv_header()

        content_for_csv.extend(dh._get_csv_data())

    write_to_csv()




