from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Load the event file from one of your directories
path = "runs/Centralized_D2QN"
event_acc = EventAccumulator(path)
event_acc.Reload()

print("Available scalar tags in your logs:")
for tag in event_acc.Tags()['scalars']:
    print(f"- '{tag}'")