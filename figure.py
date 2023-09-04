import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import yaml
import argparse
import sys

script_path = os.path.dirname(__file__).replace("\\", "/")

parser = argparse.ArgumentParser(description='Process a configuration file.')
parser.add_argument('config_path', nargs='?', default= os.path.join(script_path, "config.yaml"), help='Path to the configuration file (e.g., config.yaml)')
args = parser.parse_args()

# loading config file
def load_yaml_config(config_path):

    try:
        file = open(config_path, "r")
        config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found.")
        sys.exit(1)

cfg = load_yaml_config(args.config_path)


save_plot = cfg.get("save_plot")

model_path = os.path.join(script_path, cfg.get("model_trained_path"), "runs")

logging_dir = model_path.replace("\\", "/")

file_names = os.listdir(logging_dir)

tag_name = cfg.get("tag_name")

values, steps = [],[]

for folder in file_names:
    log_path = os.path.join(logging_dir, folder)

    event_acc = event_accumulator.EventAccumulator(log_path)
    event_acc.Reload()

    tag_data = event_acc.Scalars(tag_name)

    for event in tag_data:

        values.append(event.value)
        steps.append(event.step)
    
mid = 0
best = 100
temp = 0
# Print the values and steps
for value, step in zip(values, steps):
    print(f"Step: {step}, Value: {value}")
    mid = mid + value
    temp = value
    if temp < best:
        best = temp

model_name = "Model path: {}".format(cfg.get("model_trained_path"))
print(model_name)
print("the mean value is {}".format(mid/(len(values))))
print("the lowest value is: {}".format(best))



fig, axs = plt.subplots(figsize=(12, 4))

# Plotting in the first subplot
axs.plot(steps, values)
axs.set_title(tag_name)
axs.set_xlabel('Steps')
axs.set_ylabel('Values')
axs.grid(True)

fig.tight_layout()

if save_plot:
    plot_path = os.path.join(script_path, "figure and statistic")
    if os.path.exists(plot_path):
        pass
    else:
        os.makedirs(plot_path)
    plt.savefig(os.path.join(plot_path, cfg.get("model_trained_path") + ".png"))

plt.show(block = True)