import numpy as np


def get_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """ 3点間の角度を計算する """
    v1 = p0 - p1
    v2 = p2 - p1
    print('get_angle')
    print(p0, p1, p2)
    dot_product = np.dot(v1.flatten(), v2.flatten())
    cross_product = np.cross(v1.flatten(), v2.flatten())
    angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
    return np.degrees(angle)


if __name__ == '__main__':
    p0 = np.array([0, 0])
    p1 = np.array([1, 0])
    p2 = np.array([1, 1])

    # 角度を計算する
    angle = get_angle(p0, p1, p2)
    print(angle)
