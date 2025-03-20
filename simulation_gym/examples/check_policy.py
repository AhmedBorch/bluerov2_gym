from stable_baselines3 import PPO


if __name__ == "__main__":
    model = PPO.load("bluerov_ppo")
    print(model.policy)

    print("//////////////////////////////////////////")

    for name, param in model.policy.state_dict().items():
        print(name, param.shape)