import matplotlib.pyplot as plt
from ultrasound_bbox_env import ManualBBoxEnv

env = ManualBBoxEnv()
obs, _ = env.reset()

fig, ax = plt.subplots()
env.render(ax)

key_to_action = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "a": 4,
    "z": 5,
    "d": 6,
    "c": 7,
    "m": 8 
}


def on_key(event):
    global obs

    if event.key == "m":
        plt.close(fig)
        return

    if env.done:
        plt.close(fig)
        return

    if event.key in key_to_action:
        action = key_to_action[event.key]
        obs, reward, terminated, truncated, _ = env.step(action)

        print(f"Step: {env.steps}")
        print(f"IoU: {env.prev_iou:.4f}")
        print(f"Reward: {reward:.4f}")
        print("-" * 30)

        env.render(ax)
        fig.canvas.draw_idle()

        if terminated or truncated:
            print("Episode finished.")
            plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()