import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Reference
Best way to learn Python part 5 - OOP and bouncing balls!
http://nbviewer.ipython.org/github/cpbotha/bwtl-python-tutorials/blob/master/part%205%20-%20object%20oriented%20programming%20and%20bouncing%20balls.ipynb
"""

DIMENSION = 2
VELOCITY_WEIGHT = np.array([100, 80])
TIME_INTERVAL = 0.1
CoR = 0.95


def bounce_balls(time=128, box_size=10, ball_count=1, initial_p=None, initial_v=None):
    result = [bounce_ball(time, box_size, initial_p, initial_v) for x in range(ball_count)]
    return result


def bounce_ball(time=128, box_size=10, initial_p=None, initial_v=None):
    features = ["positions", "velocities"]
    result = np.zeros((time, len(features) * DIMENSION))

    p = initial_p
    if p is None:
        p = gen_position(box_size)
    else:
        p = np.copy(initial_p)

    v = initial_v
    if v is None:
        v = gen_velocity(box_size)
    else:
        v = np.copy(initial_v)

    t = 0
    while t < time:
        result[t] = np.copy(np.hstack((p, v)))
        p, v = __move(p, v, box_size)
        t += 1

    return result


def gen_position(box_size):
    p = np.zeros(DIMENSION)
    for i in range(DIMENSION):
        p[i] = random.uniform(0, box_size)
    return p


def gen_velocity(box_size):
    v = np.zeros(DIMENSION)
    for i in range(DIMENSION):
        v_range = box_size / (VELOCITY_WEIGHT * TIME_INTERVAL)
        v[i] = random.uniform(v_range[0], v_range[1])
    return v


def __move(p, v,  box_size):
    next_v = np.array(v)

    if p[0] <= 0:
        # hit left wall
        next_v[0] = CoR * abs(v[0])
    elif p[0] >= box_size:
        # hit right wall
        next_v[0] = - CoR * abs(v[0])

    if p[1] <= 0:
        # hit bottom wall
        next_v[1] = CoR * abs(v[1])
    elif p[1] >= box_size:
        # hit top wall
        next_v[1] = - CoR * abs(v[1])

    next_p = p + next_v * TIME_INTERVAL
    next_p = np.clip(next_p, 0, box_size)
    if next_p[0] > box_size or next_p[1] > box_size:
        print("error!")

    return next_p, next_v


def show_animation(bounce_data, box_size, name=None):
    if not isinstance(bounce_data, list):
        raise Exception("bounce_data has to be list of bounce ball")

    figure = plt.figure()
    area = figure.add_subplot(111, autoscale_on=False)
    area.xaxis.set_ticks(range(box_size + 1))
    area.yaxis.set_ticks(range(box_size + 1))
    area.grid()
    balls = []
    for b in range(len(bounce_data)):
        draw, = area.plot([], [], 'o', markersize=5)
        balls.append(draw)

    def animate(t):
        for i in range(len(bounce_data)):
            p = bounce_data[i][t][:2]
            balls[i].set_data(p[0], p[1])
        return balls

    movie = animation.FuncAnimation(figure, animate, bounce_data[0].shape[0], interval=100, blit=True)
    if name:
        try:
            movie.save("./{0}.gif".format(name), writer="imagemagick", fps=4)
        except Exception as ex:
            print("image is not saved")
            pass

    plt.show()


if __name__ == "__main__":
    _time = 128
    _box_size = 10
    _ball_count = 1
    data = bounce_balls(_time, _box_size, _ball_count)
    show_animation(data, _box_size)
