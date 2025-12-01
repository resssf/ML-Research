import json

def save_history(history, path='experiments/logs/training_history.json'):
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)

