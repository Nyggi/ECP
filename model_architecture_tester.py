from ModelBuilder import ModelBuilder
from Tools import *
from DataHandler import DataHandler
from Config import Config
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from SMF import SMF
import EvalMetrics


def hidden():
    configs = []
    x = []

    for k in range(0, 40, 10):
        for j in range(0, 300, 50):
            for i in range(1, 501, 100):
                c = Config()
                c.FEATURES = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                c.HIDDEN_LAYERS = [i]

                if j > 0:
                    c.HIDDEN_LAYERS.append(j)
                if k > 0:
                    c.HIDDEN_LAYERS.append(k)

                configs.append(c)
                x.append(i)

    return configs, x


def hidden_2():
    configs = []

    layer0 = range(1, 500, 40)
    layer1 = range(0, 250, 30)

    x = [w for w in layer0]
    y = [h for h in layer1]

    for j in layer1:
        for i in layer0:
            c = Config()
            c.HIDDEN_LAYERS = [i]

            if j > 0:
                c.HIDDEN_LAYERS.append(j)

            configs.append(c)

    return configs, x, y


def hidden_1():
    configs = []

    layer0 = range(1, 250, 10)
    x = [w for w in layer0]

    for i in layer0:
        c = Config()
        c.HIDDEN_LAYERS = [i]

        configs.append(c)

    return configs, x


# configs, x, y = hidden_2()
configs, x = hidden_1()

evaluations = []
nr = 0

for cfg in configs:
    print("Config: " + str(nr))

    total = 0
    iterations = 1

    for i in range(iterations):
        smf = SMF(cfg)

        smf.train_model()

        eval_indi, evals = smf.eval_model([EvalMetrics.mape])

        total += evals[0]

        print(f'MAPE: {evals[0]:.2f} {cfg.HIDDEN_LAYERS}')

    evaluations.append(total / iterations)

    nr += 1


if 'y' in locals():
    z = np.reshape(np.array(evaluations), (len(y), len(x)))

    fig = plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    heatplot = ax.imshow(z, cmap='BuPu')
    ax.set_xticklabels([1] + x)
    ax.set_yticklabels([0] + y)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_title("2 hidden layers")
    ax.set_xlabel('Layer0')
    ax.set_ylabel('Layer1')

    plt.colorbar(heatplot)
else:
    plt.plot(x, evaluations)

plt.show()


