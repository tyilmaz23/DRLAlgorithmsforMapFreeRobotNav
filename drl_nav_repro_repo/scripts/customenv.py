class CustomEnv(gym.Env):
    GOAL_REACHED_DIST = 0.5
    COLLISION_DIST = 0.17
    LIDAR_ANGLES = np.linspace(0, 2 * np.pi, 120, endpoint=False)

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.rng = np.random.default_rng(seed=rng.integers(1e9))

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float64),
            high=np.array([1.0, 1.0], dtype=np.float64),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1] + [-1]*120, dtype=np.float32),
            high=np.array([ 1,  1,  1,  1,  1,  1] + [ 1]*120, dtype=np.float32),
            dtype=np.float32
        )

        self.wall_thickness = 0.30
        self.max_steps = 1000
        self.D = 66e-3; self.R = self.D/2; self.L = 160e-3

        # episode state
        self.agent_pos = np.array([-1.0, -1.3, np.pi/2], dtype=np.float32)
        self.target_pos = np.array([4.0, 4.0], dtype=np.float32)
        self.distance_buffer = deque(maxlen=5)
        self.MIN_LIDAR = 0.16; self.MAX_LIDAR = 1.00

        # logs
        self.path = []
        self.walls = []
        self.dynamic_walls = []
        self.all_targets = []
        self.target_outcomes = []

        # motif
        self.episode_trace = []
        self.failure_memory = []

        # === SABİT SHAPING ===
        self.DANGER_THRESH = 0.23
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.rw = {
            "k_danger": -0.0010,  # duvara yakınlık cezası (negatif)
            "k_heading": 0.0008,  # hedefe hizalanma ödülü (pozitif)
            "k_smooth": -0.0020,  # anti-salınım: -|Δaction|^2
            "k_extra": 0.01       # final verimlilik bonusu
        }

    # --- yardımcılar ---
    def drain_failures(self):
        out, self.failure_memory = self.failure_memory, []
        return out

    def ilerikin(self, wR, wL):
        v = 0.5 * self.R * (wR + wL)
        w = (self.R / self.L) * (wR - wL)
        return v, w

    def terskin(self, v, w):
        wL = (1 / self.R) * (v - w * self.L / 2)
        wR = (1 / self.R) * (v + w * self.L / 2)
        return wL, wR

    def lidar_readings(self):
        lidar_distances = np.full(len(self.LIDAR_ANGLES), 1.00)
        distances = np.linspace(0.01, 1.00, 100)
        for i, angle in enumerate(self.LIDAR_ANGLES):
            dx = distances * np.cos(self.agent_pos[2] + angle)
            dy = distances * np.sin(self.agent_pos[2] + angle)
            points_x = self.agent_pos[0] + dx
            points_y = self.agent_pos[1] + dy
            for wall in self.walls + self.dynamic_walls:
                in_wall = (wall[0] <= points_x) & (points_x <= wall[2]) & (wall[1] <= points_y) & (points_y <= wall[3])
                if np.any(in_wall):
                    lidar_distances[i] = distances[np.argmax(in_wall)]
                    break
        return np.clip(lidar_distances, 0.16, 1.00)

    # --- map generation (12 engel) ---
    def place_dynamic_walls(self):
        self.dynamic_walls = []
        horiz_lengths = [3.0, 2.0, 0.5, 1.5, 1.25, 0.75]
        vert_lengths  = [3.0, 2.0, 1.5, 1.25, 1.0, 0.75]
        target_n = 12
        while len(self.dynamic_walls) < target_n:
            i = len(self.dynamic_walls)
            if i < 6:
                L = horiz_lengths[i];  wall = self.random_wall(L, axis='x')
            else:
                L = vert_lengths[i-6]; wall = self.random_wall(L, axis='y')
            if self.is_valid_wall_position(wall):
                self.dynamic_walls.append(wall)

    def is_valid_wall_position(self, wall):
        threshold = 0.3
        if (wall[0]-threshold <= self.agent_pos[0] <= wall[2]+threshold) and (wall[1]-threshold <= self.agent_pos[1] <= wall[3]+threshold):
            return False
        for existing_wall in self.dynamic_walls:
            if self.distance_between_walls(wall, existing_wall) < 1.0:
                return False
        return True

    def distance_between_walls(self, wall1, wall2):
        x1, y1 = (wall1[0] + wall1[2]) / 2, (wall1[1] + wall1[3]) / 2
        x2, y2 = (wall2[0] + wall2[2]) / 2, (wall2[1] + wall2[3]) / 2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def random_wall(self, length, axis='x'):
        if axis == 'x':
            x_start = self.rng.uniform(-5 + self.wall_thickness, 5 - length)
            y_pos   = self.rng.uniform(-5 + self.wall_thickness, 5 - self.wall_thickness)
            return (x_start, y_pos, x_start + length, y_pos + self.wall_thickness)
        else:
            y_start = self.rng.uniform(-5 + self.wall_thickness, 5 - length)
            x_pos   = self.rng.uniform(-5 + self.wall_thickness, 5 - self.wall_thickness)
            return (x_pos, y_start, x_pos + self.wall_thickness, y_start + length)

    # --- env API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([
            self.rng.uniform(-3, 3),
            self.rng.uniform(-3, 3),
            self.rng.uniform(0, 2 * np.pi)
        ], dtype=np.float32)
        self.place_dynamic_walls()

        threshold = 0.3
        while True:
            self.target_pos = np.array([self.rng.uniform(-4.3, 4.3), self.rng.uniform(-4.3, 4.3)], dtype=np.float32)
            if np.linalg.norm(self.agent_pos[0:2] - self.target_pos) < 3.0:
                continue
            ok = True
            for wall in self.walls + self.dynamic_walls:
                if (wall[0]-threshold <= self.target_pos[0] <= wall[2]+threshold) and (wall[1]-threshold <= self.target_pos[1] <= wall[3]+threshold):
                    ok = False; break
            if ok: break

        self.all_targets.append(self.target_pos.copy())
        self.target_outcomes.append(None)

        self.step_count = 0
        self.accumulated_distance = 0.0
        self.path = [self.agent_pos[:2].copy()]
        self.episode_trace = []
        self.prev_action = np.zeros(2, dtype=np.float32)

        distance_to_target = np.linalg.norm(self.agent_pos[0:2] - self.target_pos)
        self.distance_to_target_start = distance_to_target - 0.5
        self.distance_buffer = deque([distance_to_target] * 5, maxlen=5)

        lidar_data = self.lidar_readings()
        self.MIN_LIDAR = min(self.MIN_LIDAR, np.min(lidar_data))
        self.MAX_LIDAR = max(self.MAX_LIDAR, np.max(lidar_data))
        if self.MIN_LIDAR == self.MAX_LIDAR: self.MAX_LIDAR += 0.01
        normalized_lidar = np.clip(((lidar_data - self.MIN_LIDAR)/(self.MAX_LIDAR - self.MIN_LIDAR))*2 - 1, -1, 1)

        theta_to_target = self.calculate_theta_to_target(self.agent_pos, self.target_pos)
        sin_theta, cos_theta = np.sin(theta_to_target), np.cos(theta_to_target)
        normalized_distance = (distance_to_target) / (100)

        state = np.concatenate([[sin_theta, cos_theta, 0, 0, normalized_distance, 0.0], normalized_lidar]).astype(np.float32)
        return state, {}

    def step(self, action):
        info = {}

        # --- PRE-STEP lidar (shaping ve çarpışma için kullanıyoruz)
        pre_lidar = self.lidar_readings()
        pre_min_lidar = float(np.min(pre_lidar))

        # aksiyon -> hızlar (speed guard YOK)
        linear_velocity  = (action[0] + 1) * 0.11
        angular_velocity = action[1] * 2.84
        wL, wR = self.terskin(linear_velocity, angular_velocity)
        v, w  = self.ilerikin(wR, wL)

        # hareket
        dt = 0.2
        theta = self.agent_pos[2]
        self.agent_pos[0] += v * np.cos(theta) * dt
        self.agent_pos[1] += v * np.sin(theta) * dt
        self.agent_pos[2] = (self.agent_pos[2] + w * dt) % (2 * np.pi)
        self.agent_pos[0:2] = np.clip(self.agent_pos[0:2], -5 + self.wall_thickness, 5 - self.wall_thickness)
        self.path.append(self.agent_pos[:2].copy())

        # metrikler
        distance_to_target = np.linalg.norm(self.agent_pos[0:2] - self.target_pos)
        self.distance_buffer.append(distance_to_target)
        distance_diff = self.distance_buffer[-1] - self.distance_buffer[0]
        step_distance = v * dt
        self.accumulated_distance += step_distance

        # POST-STEP lidar
        lidar_data = self.lidar_readings()
        post_min_lidar = float(np.min(lidar_data))
        min_lidar_distance = min(pre_min_lidar, post_min_lidar)

        target_reached = distance_to_target < self.GOAL_REACHED_DIST
        collision = min_lidar_distance < self.COLLISION_DIST
        timeout = self.step_count >= self.max_steps - 0.1

        theta_to_target = self.calculate_theta_to_target(self.agent_pos, self.target_pos)

        # --- SABİT ana ödül (eski form)
        reward = self._reward(distance_to_target, target_reached, collision,
                              linear_velocity, angular_velocity, min_lidar_distance, distance_diff)

        # --- SHAPING (sabit katsayılar)
        if not (target_reached or collision or timeout):
            # duvara yakınlık
            danger_term = max(0.0, self.DANGER_THRESH - min_lidar_distance)
            reward += self.rw["k_danger"] * danger_term
            # hedefe hizalanma
            heading_term = float(np.cos(theta_to_target))  # [-1, 1]
            reward += self.rw["k_heading"] * heading_term
            # anti-salınım (Δaction^2)
            dact = float(np.sum((np.array(action, dtype=np.float32) - self.prev_action)**2))
            reward += self.rw["k_smooth"] * dact

        # --- Final verimlilik bonusu (eff) ---
        if target_reached or timeout:
            progress = max(0.0, (self.distance_to_target_start + 0.5) - distance_to_target)
            eff = progress / max(self.accumulated_distance + 1e-6, 1e-6)
            eff = min(1.0, eff)  # [0,1]
            reward += self.rw["k_extra"] * eff

        # TRACE
        self.episode_trace.append({
            "t": int(self.step_count),
            "pos": self.agent_pos[:2].tolist(),
            "theta": float(self.agent_pos[2]),
            "dist": float(distance_to_target),
            "ddiff": float(distance_diff),
            "min_lidar_pre": float(pre_min_lidar),
            "min_lidar_post": float(post_min_lidar),
            "lin_v": float(linear_velocity),
            "ang_v": float(angular_velocity),
            "guarded": False,
            "action": [float(action[0]), float(action[1])],
            "action_guarded": [float(action[0]), float(action[1])]
        })
        self.prev_action = np.array(action, dtype=np.float32)

        # normalize obs
        normalized_distance = (distance_to_target) / (100)
        normalized_linear_velocity = ((linear_velocity - 0) / (0.22 - 0)) * 2 - 1
        normalized_angular_velocity = ((angular_velocity - (-2.84)) / (2.84 - (-2.84))) * 2 - 1
        self.MIN_LIDAR = min(self.MIN_LIDAR, np.min(lidar_data))
        self.MAX_LIDAR = max(self.MAX_LIDAR, np.max(lidar_data))
        if self.MIN_LIDAR == self.MAX_LIDAR: self.MAX_LIDAR += 0.01
        normalized_lidar = np.clip(((lidar_data - self.MIN_LIDAR)/(self.MAX_LIDAR - self.MIN_LIDAR))*2 - 1, -1, 1)
        normalized_distance_diff = (distance_diff / 100)

        done = target_reached or collision or timeout
        if done:
            if target_reached:
                self.target_outcomes[-1] = 'reached'; info['reason'] = 'target_reached'
            elif collision:
                self.target_outcomes[-1] = 'collision'; info['reason'] = 'collision'
            elif timeout:
                self.target_outcomes[-1] = 'timeout'; info['reason'] = 'max_steps_reached'

            if not target_reached:
                try:
                    trace_ds = self.episode_trace[::2]
                    path_ds = self.path[::2]
                    self.failure_memory.append({
                        "reason": info.get("reason", "unknown"),
                        "trace": trace_ds,
                        "path": [p.tolist() for p in path_ds],
                        "start": self.path[0].tolist(),
                        "target": self.target_pos.tolist()
                    })
                except Exception:
                    pass

        truncated = bool(timeout)
        terminated = bool(target_reached or collision)
        self.step_count += 1

        state = np.concatenate([[np.sin(theta_to_target),
                                 np.cos(theta_to_target),
                                 normalized_linear_velocity, normalized_angular_velocity,
                                 normalized_distance, normalized_distance_diff], normalized_lidar]).astype(np.float32)
        return state, float(reward), terminated, truncated, info

    def calculate_theta_to_target(self, agent_pos, target_pos):
        delta_x = target_pos[0] - agent_pos[0]
        delta_y = target_pos[1] - agent_pos[1]
        target_theta = np.arctan2(delta_y, delta_x)
        return target_theta - agent_pos[2]

    def _reward(self, distance_to_target, target_reached, collision, linear_velocity, angular_velocity, min_lidar_distance, distance_diff):
        # Eski sabit çekirdek ödül (aynen)
        reward = 0.0
        if target_reached:
            reward += 1.0
        if collision:
            reward = -1.0
        if self.step_count >= self.max_steps-0.1:
            reward += -0.6
        if not (target_reached or collision or self.step_count >= self.max_steps):
            reward += (0.0006 * (1 / max(1e-12, 2 * distance_to_target))) \
                      - (0.00035 * (self.step_count / self.max_steps)) \
                      - (abs(angular_velocity) * 0.00018) \
                      + (-0.0004 * (distance_diff)) \
                      + (linear_velocity * 0.00022)
        return reward
