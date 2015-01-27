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
VELOCITY_WEIGHT = np.array([10, 8])
TIME_INTERVAL = 0.1
CoR = 0.95


def bounce_ball(time=128, box_size=10, ball_count=1, initial_p=None, initial_v=None):
    positions = []
    velocities = []

    p = initial_p
    if p is None:
        p = gen_position(box_size, ball_count)
    else:
        p = np.copy(initial_p)

    v = initial_v
    if v is None:
        v = gen_velocity(box_size, ball_count)
    else:
        v = np.copy(initial_v)

    t = 0
    while t < time:
        positions.append([])
        velocities.append([])
        for i in range(ball_count):
            positions[t].append(np.copy(p[i]))
            velocities[t].append(np.copy(v[i]))
            p[i], v[i] = __move(p[i], v[i], box_size)

        t += 1

    return positions, velocities


def gen_position(box_size, ball_count):
    p = []
    for i in range(ball_count * DIMENSION):
        p.append(random.uniform(0, box_size))

    return np.array(p).reshape(ball_count, DIMENSION)


def gen_velocity(box_size, ball_count):
    v = []
    # 2 means 2-dimension (position is expressed by x, y)
    for i in range(ball_count * DIMENSION):
        v_range = box_size / (VELOCITY_WEIGHT * TIME_INTERVAL)
        v.append(random.uniform(v_range[0], v_range[1]))

    return np.array(v).reshape(ball_count, DIMENSION)


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

    return next_p, next_v


def show_animation(bounce_data, box_size, name=None):
    if len(bounce_data) == 0:
        return False

    figure = plt.figure()
    area = figure.add_subplot(111, autoscale_on=False, xlim=box_size, ylim=box_size)
    area.grid()
    balls = []
    for b in range(len(bounce_data[0])):
        draw, = area.plot([], [], 'o', markersize=5)
        balls.append(draw)

    def animate(t):
        ps = bounce_data[t]
        for i, p in enumerate(ps):
            balls[i].set_data(p[0], p[1])
        return balls

    movie = animation.FuncAnimation(figure, animate, np.arange(len(bounce_data)), interval=100, blit=True)
    if name:
        try:
            movie.save("./{0}.gif".format(name), writer="imagemagick", fps=4)
        except Exception as ex:
            print("image is not saved")
            pass

    plt.show()


if __name__ == "__main__":
    _time = 40
    _box_size = 10
    _ball_count = 1
    data, _ = bounce_ball(_time, _box_size, _ball_count)
    show_animation(data, _box_size)
