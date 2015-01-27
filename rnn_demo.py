import sys
from statistics import median
import numpy as np
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.structure import RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import data.ball_data as ball_data


BALL_COUNT = 1
BOX_SIZE = 10


def predict_ball(hidden_nodes, is_elman=True, training_data=5000, training_batch=-1, predict_count=64):
    batch_size = training_data
    if training_batch > 0:
        if training_data < training_batch:
            raise Exception("training count have to be greater than training batch size")
        else:
            batch_size = training_batch

    # build rnn
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(4, name="i"))
    n.addModule(BiasUnit("b"))
    n.addModule(SigmoidLayer(hidden_nodes, name="h"))
    n.addOutputModule(LinearLayer(4, name="o"))

    n.addConnection(FullConnection(n["i"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["o"]))
    n.addConnection(FullConnection(n["h"], n["o"]))

    if is_elman:
        # Elman (hidden->hidden)
        n.addRecurrentConnection(FullConnection(n["h"], n["h"]))
    else:
        # Jordan (out->hidden)
        n.addRecurrentConnection(FullConnection(n["o"], n["h"]))

    n.sortModules()

    # make training data
    initial_v = ball_data.gen_velocity(BOX_SIZE, BALL_COUNT)
    training_ds = SupervisedDataSet(4, 4)
    p_all = []

    for b in range(training_data // batch_size):
        p_s, v_s = ball_data.bounce_ball(batch_size + 1, BOX_SIZE, BALL_COUNT, None, initial_v=initial_v)
        p_all += p_s
        p_s = __normalize(p_s)
        v_s = __normalize(v_s)
        for i in range(batch_size):
            # from current, predict next
            p_in = p_s[i][0].tolist() + v_s[i][0].tolist()
            p_out = p_s[i + 1][0].tolist() + v_s[i + 1][0].tolist()
            training_ds.addSample(p_in, p_out)

    # training network
    trainer = BackpropTrainer(n, training_ds)
    err1 = trainer.train()

    # predict
    p_avg = np.average(np.array(p_all), axis=0)
    p_std = np.std(np.array(p_all), axis=0)
    initial_p = ball_data.gen_position(BOX_SIZE, BALL_COUNT)
    predicts = []
    next_pv = initial_p[0].tolist() + initial_v[0].tolist()
    next_position = initial_p[0].tolist()

    n.reset()
    for i in range(predict_count):
        predicts.append([np.array(next_position)])
        next_pv = n.activate(next_pv)
        next_position = (np.array(next_pv[:2]) * p_std + p_avg).tolist()[0]

    real_p, _ = ball_data.bounce_ball(predict_count, BOX_SIZE, BALL_COUNT, initial_p, initial_v)

    err2 = __calc_diff(predicts, real_p)

    return predicts, real_p, err1, err2


def __normalize(arr):
    a = np.array(arr)
    normalized = (a - np.average(a, axis=0)) / np.std(a, axis=0)
    return normalized


def __calc_diff( predicts, results):
    diffs = []
    for i, v in enumerate(results):
        diff = 0
        for j, b in enumerate(results[i]):
            diff += np.sqrt(np.sum((predicts[i][j] - results[i][j]) ** 2))
        diffs.append(diff)

    return np.mean(diffs)


def eval_model(min_hidden, max_hidden, is_elman=True, step=10, training_data=5000, trial_run=10):
    for h in range(min_hidden, max_hidden + step, step):
        training_e = []
        test_e = []

        for i in range(trial_run):
            p, r, e1, e2 = predict_ball(h, is_elman, training_data)
            print("{0}\t{1}\t{2}".format(h, e1, e2))
            training_e.append(e1)
            test_e.append(e2)

        print("{0}\t{1}\t{2}".format(h, median(training_e), median(test_e)))


def main(is_elman=True):
    # evaluate model by changing hidden layer
    # eval_model(4, 20, True, step=1)

    nodes = 4
    p, r, e1, e2 = predict_ball(nodes, is_elman=is_elman, training_data=20000, training_batch=5000)
    print("training error:{0}, test error:{1}".format(e1, e2))
    ball_data.show_animation(r, BOX_SIZE)
    ball_data.show_animation(p, BOX_SIZE)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "E":
        main(True)
    else:
        main(False)
