import time
import json
import re
import numpy as np
from llama_cpp import Llama
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from bluerov2_gym.envs.bluerov_env import BlueRov

# ======================
# 1) SHAPE‐GENERATORS
# ======================

def generate_circle_waypoints(center=(0.0, 0.0), radius=1.0, num_points=36, z=0.0):
    cx, cy = center
    waypoints = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        waypoints.append([float(x), float(y), float(z)])
    if num_points > 0:
        waypoints.append([float(cx + radius), float(cy + 0.0), float(z)])
    return waypoints

def generate_polygon_waypoints(vertices_xy, z=0.0, close_loop=True):
    waypoints = [[float(x), float(y), float(z)] for (x, y) in vertices_xy]
    if close_loop and len(vertices_xy) > 0:
        first = vertices_xy[0]
        waypoints.append([float(first[0]), float(first[1]), float(z)])
    return waypoints

def parse_shape_command(command: str):
    text = command.lower()

    # CIRCLE
    if "circle" in text:
        radius = 1.0
        cx, cy = 0.0, 0.0
        num_pts = 36
        m = re.search(r"radius\s*=\s*([-\d\.]+)", text)
        if m:
            radius = float(m.group(1))
        m2 = re.search(r"center\s*=\s*\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?", text)
        if m2:
            cx, cy = float(m2.group(1)), float(m2.group(2))
        m3 = re.search(r"points\s*=\s*(\d+)", text)
        if m3:
            num_pts = int(m3.group(1))
        return generate_circle_waypoints(center=(cx, cy), radius=radius, num_points=num_pts, z=0.0)

    # SQUARE
    if "square" in text:
        side = 1.0
        cx, cy = 0.0, 0.0
        m = re.search(r"side\s*=\s*([-\d\.]+)", text)
        if m:
            side = float(m.group(1))
        m2 = re.search(r"center\s*=\s*\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?", text)
        if m2:
            cx, cy = float(m2.group(1)), float(m2.group(2))
        half = side / 2.0
        verts = [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
        return generate_polygon_waypoints(verts, z=0.0)

    # RECTANGLE
    if "rectangle" in text:
        width, height = 1.0, 1.0
        cx, cy = 0.0, 0.0
        m = re.search(r"width\s*=\s*([-\d\.]+)", text)
        if m:
            width = float(m.group(1))
        m2 = re.search(r"height\s*=\s*([-\d\.]+)", text)
        if m2:
            height = float(m2.group(1))
        m3 = re.search(r"center\s*=\s*\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?", text)
        if m3:
            cx, cy = float(m3.group(1)), float(m3.group(2))
        half_w, half_h = width / 2.0, height / 2.0
        verts = [
            (cx - half_w, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
        return generate_polygon_waypoints(verts, z=0.0)

    # TRIANGLE
    if "triangle" in text:
        pts = re.findall(r"p[123]\s*=\s*\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?", text)
        if len(pts) == 3:
            verts = [(float(x), float(y)) for x, y in pts]
            return generate_polygon_waypoints(verts, z=0.0)

    # POLYGON (generic)
    if "polygon" in text:
        m = re.search(r"polygon\s*=\s*\[\s*(\([^\]]+\))\s*\]", text)
        if m:
            inside = m.group(1)
            pts = re.findall(r"\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?", inside)
            if len(pts) >= 3:
                verts = [(float(x), float(y)) for x, y in pts]
                return generate_polygon_waypoints(verts, z=0.0)

    return None


# ==============================================================
# 2) ENVIRONMENT CONFIGURATION
# ==============================================================

class EnhancedBlueRov(BlueRov):
    """Adds time penalty to reward function"""
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        reward -= 0.1 * self.current_step
        return obs, reward, done, truncated, info

def create_env(render_mode="human"):
    """Environment factory with enhanced rewards"""
    raw_env = EnhancedBlueRov(
        render_mode=render_mode,
        goal_tolerance=0.3,
        max_episode_steps=800
    )
    return TimeLimit(raw_env, max_episode_steps=800)


# ==============================================================
# 3) LLM LOADER & JSON PARSING
# ==============================================================

def load_llm(model_path: str):
    """ROCm-optimized LLM loader for AMD 6950XT"""
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=16,
        n_gpu_layers=40,
        main_gpu=0,
        tensor_split=[16],
        hip=True,
        verbose=False
    )

def extract_waypoints(text: str):
    """Robust JSON parsing with markdown handling"""
    cleaned = re.sub(
        r"```(?:json)?\n(.*?)```",
        r"\1",
        text,
        flags=re.DOTALL
    ).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "waypoints" in obj and isinstance(obj["waypoints"], list):
            return obj["waypoints"]
    except json.JSONDecodeError:
        pass

    match = re.search(r'"waypoints"\s*:\s*(\[[\s\S]*\])', cleaned)
    if match:
        array_text = match.group(1)
        try:
            wps = json.loads(array_text)
            if isinstance(wps, list):
                return wps
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid waypoints found in LLM response")


def query_waypoints(llm, command: str, max_retries=3):
    """
    First check if `command` mentions a standard 2D shape. If yes, generate
    waypoints directly (z=0). Otherwise, call the LLM as before.
    """
    # 1) See if the command is a recognized shape
    shape_waypoints = parse_shape_command(command)
    if shape_waypoints is not None:
        return shape_waypoints

    # 2) Otherwise, fall back on the LLM prompt & JSON-parsing logic
    system_prompt = (
        "You are a waypoint generator. You must respond with valid JSON only, no extra text or markdown. "
        "Output exactly:\n"
        "{\n"
        "  \"waypoints\": [[x1, y1, z1], [x2, y2, z2], …]\n"
        "}\n"
        "Rules:\n"
        "- x,y ∈ [-6.0, 6.0], z ∈ [-6.0, 0.0]\n"
        "- All numbers must be floats\n"
        "Example (valid JSON!): {\"waypoints\": [[2.0, 2.0, -3.0], [2.0, -2.0, -3.0]]}"
    )

    for attempt in range(max_retries):
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            raw_text = response['choices'][0]['message']['content']
            print("RAW LLM OUTPUT:")
            print(raw_text)
            wps = extract_waypoints(raw_text)

            # Only force z=0 if the command does NOT imply depth:
            if not re.search(r"\b(z|depth|dive|down|up|ascend)\b", command, flags=re.IGNORECASE):
                wps = [[float(x), float(y), 0.0] for x, y, _ in wps]

            return wps

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")

    raise RuntimeError(f"Failed after {max_retries} attempts")


# ==============================================================
# 4) MISSION EXECUTION (NO LOOP‐CLOSURE)
# ==============================================================

def execute_mission(policy, env, waypoints, step_delay=0.1):
    raw_env = env.env

    # 1) Start from a fresh reset (initial state)
    obs, _ = env.reset()

    # 2) ←– We have REMOVED the automatic “close‐loop” here 
    # if not np.allclose(waypoints[0], waypoints[-1]):
    #     waypoints.append(waypoints[0].copy())

    total_steps = 0  # optional: track cumulative steps

    # 3) Iterate over each “leg” from waypoint[i] → waypoint[i+1]
    for idx in range(len(waypoints) - 1):
        current_wp = waypoints[idx]
        next_wp = waypoints[idx + 1]

        raw_env.goal = np.array(next_wp, dtype=np.float32)
        if raw_env.render_mode == "human":
            raw_env.renderer.update_waypoint(raw_env.goal)

        print(f"→ Leg {idx+1}: {current_wp} → {next_wp}")

        done = truncated = False
        while not (done or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            raw_env.step_sim()
            total_steps += 1
            time.sleep(step_delay)

        success = info.get("is_success", False)
        print(f"   {'SUCCESS' if success else 'FAILURE'} after {raw_env.current_step} steps")

        # Reset only the step counter so that rewards/penalties start fresh for next leg
        raw_env.current_step = 0

    print(f"Total steps taken in mission: {total_steps}")


# ==============================================================
# 5) MAIN EXECUTION
# ==============================================================

if __name__ == "__main__":
    CONFIG = {
        "model_path": "ppo_bluerov_phase2.zip",
        "llm_path": "/home/elex/aiproject/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf",
        "render_mode": "human",
        "preload_sec": 7.0,
        "step_delay": 0.1
    }

    print("Loading PPO policy...")
    policy = PPO.load(CONFIG["model_path"], device="cpu")

    print("Creating environment...")
    env = create_env(CONFIG["render_mode"])

    print("Initializing LLM...")
    llm = load_llm(CONFIG["llm_path"])

    if CONFIG["render_mode"] == "human":
        print(f"Waiting {CONFIG['preload_sec']}s for MeshCat initialization...")
        time.sleep(CONFIG["preload_sec"])

    try:
        while True:
            cmd = input("\nEnter navigation command (or 'quit'): ").strip()
            if cmd.lower() in ("quit", "exit"):
                break

            print("\nGenerating waypoints...")
            try:
                waypoints = query_waypoints(llm, cmd)
                print(f"Validated waypoints: {waypoints}")
                print("Executing mission...")
                execute_mission(policy, env, waypoints, CONFIG["step_delay"])
            except Exception as e:
                print(f"Mission failed: {str(e)}")
    finally:
        env.close()
        print("\nSimulation closed")
