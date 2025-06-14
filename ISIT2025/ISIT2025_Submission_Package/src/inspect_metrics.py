import numpy as np

metrics = np.load('results/basic_localization/metrics.npy', allow_pickle=True)
print("metrics.npy contents:")
print(metrics)

if hasattr(metrics, 'item'):
    print("\nAs dict:")
    print(metrics.item()) 