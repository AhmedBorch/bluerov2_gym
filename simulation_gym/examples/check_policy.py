from stable_baselines3 import PPO
from torch import onnx
import torch
import torch.nn as nn

class MLPOnly(nn.Module):
    def __init__(self, mlp_extractor):
        super().__init__()
        self.mlp = mlp_extractor.policy_net

    def forward(self, x):
        return self.mlp(x)



class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, *inputs):
        # Create the input dict in the same order as expected
        keys = ['omega', 'target_angle', 'target_x', 'target_y', 'target_z',
                'theta', 'vx', 'vy', 'vz', 'x', 'y', 'z']
        obs = {k: v for k, v in zip(keys, inputs)}

        # Debugging: Print the result of self.policy(obs, deterministic=True)
        result = self.policy(obs, deterministic=True)
        print("Policy output:", result)  # Debug print statement

        # If it returns multiple values (actions, log_probs, values, etc.), unpack them
        actions, log_probs, value = result  # Update this line based on the actual return structure of self.policy

        return actions

if __name__ == "__main__":
    model = PPO.load("examples/2mio_donut/bluerov_ppo_scratch2")
    print(model.policy)
    policy = model.policy
    dummy_inputs = tuple(torch.zeros((1, 1), dtype=torch.float32) for _ in range(13))

    wrapped_policy = PolicyWrapper(model.policy)
    wrapped_policy.eval()  # Set to eval mode

    print("Testing Policy Output:")
    result = wrapped_policy(*dummy_inputs)
    print(result)

    torch.onnx.export(
        wrapped_policy,
        dummy_inputs,
        "ppo_policy.onnx",
        input_names=['omega', 'target_angle', 'target_x', 'target_y', 'target_z',
                    'theta', 'vx', 'vy', 'vz', 'x', 'y', 'z'],
        output_names=["actions"],
        dynamic_axes={name: {0: 'batch_size'} for name in [
            'omega', 'target_angle', 'target_x', 'target_y', 'target_z',
            'theta', 'vx', 'vy', 'vz', 'x', 'y', 'z']},
        opset_version=11
    )
    # dummy = torch.zeros((1, 12))  # 12 flattened features
    # model = MLPOnly(model.policy.mlp_extractor)

    # torch.onnx.export(model, dummy, "mlp_policy.onnx")
    print("//////////////////////////////////////////")

    for name, param in model.policy.state_dict().items():
        print(name, param.shape)