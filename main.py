# Import:
import math as mat
import h5py
from pynput.keyboard import Key, Listener
import gym
import time
import queue
import threading
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


# Image operation
# --------------------------------------------------------------------------------
# This function convert RGB to Grey
def rgb_to_grey(im):
    return np.dot(im, [0.2126, 0.7151, 0.0722])


# This function image crop and resize
def cv2_resize_image(image, resized_shape=(82, 82), method='crop', crop_offset=8):
    height, width = image.shape
    resized_height, resized_width = resized_shape
    if method == 'crop':
        h = int(round(float(height) * resized_width / width))
        resized = cv2.resize(image, (resized_width, h), interpolation=cv2.INTER_LINEAR)
        crop_y_cutoff = h - crop_offset - resized_height
        cropped = resized[crop_y_cutoff:crop_y_cutoff + resized_height, :]
        return np.asarray(cropped, dtype=np.uint8)
    elif method == 'scale':
        return np.asarray(cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR),
                          dtype=np.uint8)
    elif method == 'none':
        return np.asarray(image, dtype=np.uint8)
    else:
        raise ValueError('Nie ter resize method!')


def image_operation(observation, name=0, save=True):
    # Generate image
    img = rgb_to_grey(observation)
    img = cv2_resize_image(img, method='none')
    if save:
        img = Image.fromarray(img)
        img.save('{}.jpg'.format(name))
    else:
        return img


# Main control function
def keyboard(queue):
    def on_press(key):
        if key == Key.esc:
            queue.put(-1)
        elif key == Key.space:
            queue.put(ord(' '))
        else:
            key = str(key).replace("'", '')
            if key in ['w', 'a', 's', 'd']:
                queue.put(ord(key))

    def on_release(key):
        if key == Key.esc:
            return False

    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# Game
# --------------------------------------------------------------------------------
def positions(observation, observation_previous, distance, distance_previous):
    """
    This function compares two images (previous and actual) and following the changes returns the ball's position
    :param observation: actual image
    :param observation_previous: previous image
    :param distance: actual distance between the ball and the space
    :param distance_previous: previous distance between the ball and the space
    :return: actual distance and reward
    """
    # RGB to Grey
    observation = rgb_to_grey(observation)
    observation_previous = rgb_to_grey(observation_previous)
    # -----------------------------------------------------------------------------
    # Position on object
    bol_position = np.not_equal(observation[110:-21, :], observation_previous[110:-21, :])
    space_position = np.not_equal(observation[-21:, :], observation_previous[-21:, :])
    # -----------------------------------------------------------------------------
    bol_position = np.argwhere(bol_position)
    space_position = np.argwhere(space_position)
    # -----------------------------------------------------------------------------
    if bol_position.size != 0:
        mean_index_col = 110 + int(np.mean(bol_position[:, 0]))
        mean_index_row = int(np.mean(bol_position[:, 1]))

        if space_position.size != 0:
            space_mean_index_col = 189 + int(np.mean(space_position[:, 0]))
            space_mean_index_row = int(np.mean(space_position[:, 1]))
            distance = mat.sqrt((mean_index_row - space_mean_index_row) ** 2 + (
                    mean_index_col - space_mean_index_col) ** 2)
            distance = int(distance)

            # Create reward
            reward = (distance_previous - distance)
            if reward > 0:
                reward = 2 / reward
            elif reward < 0:
                reward = -2
            elif distance < 9:
                reward = 4
            else:
                if distance_previous == 160:
                    reward = -3
                else:
                    reward = -1
        else:
            reward = -1
            distance = distance_previous
        return distance, float(reward)
    else:
        # Return  reward = zero
        return distance, -1


def start_game(queue):
    # Create game
    atari = gym.make('Breakout-v0')
    # Control index
    key_to_act = atari.env.get_keys_to_action()
    key_to_act = {k[0]: a for k, a in key_to_act.items() if len(k) > 0}
    # Observation
    observation = atari.reset()
    # Crop and save observation(image)
    image_operation(observation)

    # Loop
    action = 0
    while action != -1:
        atari.render()
        action = 0 if queue.empty() else queue.get(block=False)
        if action == -1:
            break
        action = key_to_act.get(action, 0)
        observation, reward, done, _ = atari.step(action)
        if done:
            print('Game finished!')
            break
        time.sleep(0.05)


# Learning
def q_learning_game(queue, open_h5=True):
    atari = gym.make('Breakout-v0')  # Create atar game
    size = atari.observation_space.shape

    if open_h5:
        with h5py.File('dataset.h5', 'r') as hf:
            Q = hf['dataset'][:]
    else:
        Q = np.zeros([250, atari.action_space.n])

    key_to_act = atari.env.get_keys_to_action()
    key_to_act = {k[0]: a for k, a in key_to_act.items() if len(k) > 0}
    observation = atari.reset()

    # Q-learning parameter
    eta = .01
    gma = .75
    epic = 15000

    # Q-learning Algorithm
    for i in tqdm(range(epic)):
        # Reset environment
        observation_previous = atari.reset()
        ob_pr = (observation_previous, observation_previous)
        # The Q-Table learning algorithm
        LivesMax = 5
        done = False

        s = 0
        distance = 160
        _distance_ = 160
        j = 0
        a = 1
        while j < 400:
            j += 1
            atari.render()
            # Control index
            if j > 0:
                a = np.argmax(Q[s, :])  #
            # Nex atari step
            observation, reward, done, lives = atari.step(a)
            #
            _s_, add_ = positions(observation, ob_pr[0], distance, _distance_)

            if LivesMax != lives['ale.lives']:
                reward = -1

            # Q[s, a] = Q[s, a] + eta * ( add_ + gma * np.max(Q[_s_, :]) - Q[s, a])

            s = _s_

            ob_pr = (ob_pr[1], observation)

            _distance_ = _s_

            LivesMax = lives['ale.lives']
            if done:
                break
            time.sleep(0.05)
    # Save file
    with h5py.File('dataset.h5', 'w') as hf:
        hf.create_dataset("dataset", data=Q)

    print("END")


if __name__ == '__main__':
    # Zmien start_game na q_learning_game zebys zaczac uczyc,
    queue = queue.Queue(maxsize=10)
    game = threading.Thread(target=q_learning_game, args=(queue,))
    game.start()
    keyboard(queue)
