import matplotlib.pyplot as plt

# Example C-index values per epoch (replace with your actual logs)
epochs = list(range(1, 31))  # 30 epochs
train_cindex = [0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                0.90, 0.91, 0.915, 0.918, 0.922, 0.925, 0.926, 0.925, 0.924, 0.923,
                0.922, 0.920, 0.918, 0.916, 0.915, 0.914, 0.913, 0.912, 0.911, 0.910]

val_cindex = [0.68, 0.70, 0.71, 0.715, 0.718, 0.72, 0.722, 0.724, 0.726, 0.726,
              0.725, 0.723, 0.722, 0.721, 0.720, 0.719, 0.718, 0.717, 0.716, 0.715,
              0.714, 0.713, 0.712, 0.711, 0.710, 0.709, 0.708, 0.707, 0.706, 0.705]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_cindex, label='Training C-index', marker='o')
plt.plot(epochs, val_cindex, label='Validation C-index', marker='s')
plt.xlabel("Epoch")
plt.ylabel("C-index")
plt.title("Training and Validation C-index over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig("c_index_trend_chart.png")
plt.show()
