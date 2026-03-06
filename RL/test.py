# test_visualize.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

# ----------------------------
# 블럭 시각화 함수
# ----------------------------
def plot_block(ax, block, size):
    """
    block: [x, y, z, roll, pitch, yaw]
    size: [sx, sy, sz]
    """
    x, y, z, roll, pitch, yaw = block
    sx, sy, sz = size
    # 블럭의 8개 꼭짓점
    corners = np.array([
        [0, 0, 0],
        [sx, 0, 0],
        [sx, sy, 0],
        [0, sy, 0],
        [0, 0, sz],
        [sx, 0, sz],
        [sx, sy, sz],
        [0, sy, sz]
    ])
    # 회전 적용
    r = R.from_euler('xyz', [roll, pitch, yaw])
    rotated_corners = r.apply(corners)
    # 위치 이동
    rotated_corners += np.array([x, y, z])
    # 면 구성
    faces = [
        [rotated_corners[j] for j in [0,1,2,3]],
        [rotated_corners[j] for j in [4,5,6,7]],
        [rotated_corners[j] for j in [0,1,5,4]],
        [rotated_corners[j] for j in [2,3,7,6]],
        [rotated_corners[j] for j in [1,2,6,5]],
        [rotated_corners[j] for j in [0,3,7,4]]
    ]
    pc = Poly3DCollection(faces, facecolors='cyan', edgecolors='k', alpha=0.7)
    ax.add_collection3d(pc)

# ----------------------------
# 더미 객체 생성
# ----------------------------
def generate_random_objects(num_objects):
    objects = []
    for _ in range(num_objects):
        xsize = np.random.randint(20, 60)
        ysize = np.random.randint(20, 60)
        zsize = np.random.randint(10, 40)
        durability = np.random.randint(1, 6)
        objects.append({
            "size": (xsize, ysize, zsize),
            "durability": durability
        })
    return objects

num_objects = 5
objects = generate_random_objects(num_objects)

# ----------------------------
# 모델에서 나온 결과 형식 [object_index, pos(x,y,z), rotation(roll,pitch,yaw)]
# 여기서는 더미 데이터
# ----------------------------
results = []
for i, obj in enumerate(objects):
    x = np.random.uniform(0, 200)
    y = np.random.uniform(0, 200)
    z = np.random.uniform(0, 200)
    roll = np.random.uniform(0, np.pi/4)
    pitch = np.random.uniform(0, np.pi/4)
    yaw = np.random.uniform(0, np.pi/4)
    results.append([i, [x, y, z], [roll, pitch, yaw]])

# ----------------------------
# 시각화
# ----------------------------
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for res in results:
    idx, pos, rot = res
    size = objects[idx]["size"]
    plot_block(ax, pos + rot, size)

ax.set_xlim(0, 250)
ax.set_ylim(0, 250)
ax.set_zlim(0, 250)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Block Placement Visualization')
plt.show()
