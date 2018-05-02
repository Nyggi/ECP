from ModelBuilder import ModelBuilder
from Tools import *
from DataHandler import DataHandler
from Config import SingleConfig
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


def week_configs():
    configs = []
    x = []

    for i in range(1, 20):
        c = SingleConfig()
        c.FEATURES = [1, 0, 0]

        c.WEEKS = i
        configs.append(c)
        x.append(i)

    return configs, x


def hours_configs():
    configs = []
    x = []

    for i in range(1, 600, 30):
        c = SingleConfig()
        c.FEATURES = [1, 1, 0]
        c.WEEKS = 2
        c.HOURS_PAST = i

        configs.append(c)
        x.append(i)

    return configs, x


def hours_weeks_configs():
    configs = []
    x = [w for w in range(0, 14)]
    y = [h for h in range(0, 300, 50)]

    for i in x:
        for j in y:
            c = SingleConfig()
            c.EPOCHS = 500
            c.WEEKS = i
            c.HOURS_PAST = j
            configs.append(c)

    return configs, x, y


def specific_hidden():
    configs = []
    x = []
    counter = 0

    for i in range(5, 100, 10):
        for l in range(50, 500,150):
            c = SingleConfig()
            c.EPOCHS = l
            c.FEATURES = [0, 0, 0, 1]
            c.HIDDEN_LAYERS = [i]

            configs.append(c)
            x.append(counter)
            counter += 1

    return configs, x


def weeks_hours_hidden():
    configs = []
    x = []

    for i in range(20, 100, 5):
        c = SingleConfig()
        c.FEATURES = [1, 1, 0]
        c.WEEKS = 7
        c.HIDDEN_LAYERS = [100, i]

        configs.append(c)
        x.append(i)

    return configs, x


configs, x = specific_hidden()
# configs, x = week_configs()
# configs, x = hours_configs()
# configs, x, y = hours_weeks_configs()

evaluations = []
nr = 1

for cfg in configs:
    print("Config: " + str(nr) + "/" + str(len(configs)))

    if cfg.WEEKS == 0 and cfg.HOURS_PAST == 0:
        evaluations.append(0)
        nr += 1
        continue

    dh = DataHandler(cfg)

    INPUT_SHAPE = (len(dh.train_input[0]), )

    total = 0
    iterations = 3

    for i in range(iterations):
        mb = ModelBuilder(cfg, INPUT_SHAPE)

        model = mb.nn_w()

        # print("Fitting model")
        model.fit(np.array(dh.train_input), np.array(dh.train_labels), epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)

        # print("------------------Evaluation-------------------")
        evaluation = model.evaluate(np.array(dh.eval_input), np.array(dh.eval_labels), cfg.BATCH_SIZE, verbose=0)

        total += evaluation[1]

        print(f'MAPE: {evaluation[1]:.2f}\n')

    evaluations.append(total / iterations)

    nr += 1

"""
evaluations[0] = evaluations[1]

z = np.reshape(evaluations, (len(x), len(y)))

fig = plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
heatplot = ax.imshow(z, cmap='BuPu')
ax.set_xticklabels([1] + y)
ax.set_yticklabels([0] + x)

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_title("Weeks x Hours")
ax.set_xlabel('Hours')
ax.set_ylabel('Weeks')

plt.colorbar(heatplot)
"""

plt.plot(x, evaluations)
plt.show()


