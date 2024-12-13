import mujoco
import mujoco.viewer
import numpy as np
import pyquaternion as pyq

MOCAP_INDEX = 0


def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def key_callback_data(key, data):
    """
    Callback for key presses but with data passed in
    :param key: Key pressed
    :param data:  MjData object
    :return: None
    """
    global MOCAP_INDEX
    if key == 265:  # Up arrow
        data.mocap_pos[MOCAP_INDEX, 2] += 0.01
    elif key == 264:  # Down arrow
        data.mocap_pos[MOCAP_INDEX, 2] -= 0.01
    elif key == 263:  # Left arrow
        data.mocap_pos[MOCAP_INDEX, 0] -= 0.01
    elif key == 262:  # Right arrow
        data.mocap_pos[MOCAP_INDEX, 0] += 0.01
    elif key == 61:  # +
        data.mocap_pos[MOCAP_INDEX, 1] += 0.01
    elif key == 45:  # -
        data.mocap_pos[MOCAP_INDEX, 1] -= 0.01
    elif key == 260:  # Insert
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [1, 0, 0], 10)
    elif key == 261:  # Home
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [1, 0, 0], -10)
    elif key == 268:  # Home
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 1, 0], 10)
    elif key == 269:  # End
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 1, 0], -10)
    elif key == 266:  # Page Up
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 0, 1], 10)
    elif key == 267:  # Page Down
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 0, 1], -10)
    elif key == 344:  # Right shift
        MOCAP_INDEX = (MOCAP_INDEX + 1) % len(data.mocap_quat)
    else:
        print(key)


def main():
    # Load the mujoco model basic.xml
    model = mujoco.MjModel.from_xml_path("gym_so100/assets/bimanual_transfer_cube.xml")
    data = mujoco.MjData(model)

    def key_callback(key):
        key_callback_data(key, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
