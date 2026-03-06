import numpy as np

class PackingEnv:

    def __init__(self, basket_size, grid=40):
        self.bx, self.by, self.bz = basket_size
        self.grid = grid
        self.heightmap = np.zeros((grid, grid))
        self.objects = []
        self.index = 0
        self.unit_x = self.bx / self.grid
        self.unit_y = self.by / self.grid

    def set_objects(self, objects):
        self.objects = objects

    def reset(self):
        self.heightmap[:] = 0
        self.index = 0
        return self.get_state()

    def get_state(self):
        if self.index >= len(self.objects):
            return np.zeros(3)
        obj = self.objects[self.index]
        x, y, z = obj["size"]
        return np.array([x, y, z]) / 100

    def stable_score(self, x, y, w, h, z):
        region = self.heightmap[x:x+w, y:y+h]
        support = np.sum(region >= z*0.8)
        return support / (w*h + 1e-5)

    def step(self, action):
        if self.index >= len(self.objects):
            return self.get_state(), 0, True, {}

        obj = self.objects[self.index]
        xsize, ysize, zsize = obj["size"]
        durability = obj["durability"]

        # -----------------------------
        # 1. 회전 90도 단위로 제한
        # -----------------------------
        roll_idx = int(action[3] * 4) % 4
        pitch_idx = int(action[4] * 4) % 4
        yaw_idx = int(action[5] * 4) % 4

        roll = roll_idx * 90
        pitch = pitch_idx * 90
        yaw = yaw_idx * 90

        # 회전에 따라 x,y,z swap 가능
        dims = np.array([xsize, ysize, zsize])
        if roll % 180 == 90:
            dims[[1,2]] = dims[[2,1]]
        if pitch % 180 == 90:
            dims[[0,2]] = dims[[2,0]]
        if yaw % 180 == 90:
            dims[[0,1]] = dims[[1,0]]
        w, h, zsize = dims.astype(int)

        # -----------------------------
        # 2. 위치 선택
        # -----------------------------
        gx = int(action[0] * (self.grid - w))
        gy = int(action[1] * (self.grid - h))
        gx = max(0, min(self.grid - w, gx))
        gy = max(0, min(self.grid - h, gy))

        # -----------------------------
        # 3. 안정성 체크
        # -----------------------------
        base_z = np.max(self.heightmap[gx:gx+w, gy:gy+h])
        stable = self.stable_score(gx, gy, w, h, base_z)
        if stable < 0.2:   # 안정성 기준
            reward = -0.1
            done = self.index >= len(self.objects)
            self.index += 1
            return self.get_state(), reward, done, {}

        # -----------------------------
        # 4. 블록 배치
        # -----------------------------
        z_bottom = base_z
        self.heightmap[gx:gx+w, gy:gy+h] = z_bottom + zsize

        # -----------------------------
        # 5. Reward 계산
        # -----------------------------
        obj_volume = xsize * ysize * zsize
        basket_volume = self.bx * self.by * self.bz
        reward = obj_volume / basket_volume
        reward += stable * 0.5
        reward += durability * 0.3 * (1 - z_bottom/self.bz)
        reward -= (z_bottom/self.bz) * 0.2

        self.index += 1
        done = self.index >= len(self.objects)
        return self.get_state(), reward, done, {"pos": (gx, gy, z_bottom), "rotation": (roll, pitch, yaw)}