import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import torch
from typeformar.dataset_generation.batch_video_preparation import (
    read_pickle_files,
    read_from_file,
    list_pickle_files,
    data_dir,
)

key_dict = {
    57: "_",
    30: "A",
    48: "B",
    46: "C",
    32: "D",
    18: "E",
    33: "F",
    34: "G",
    35: "H",
    23: "I",
    36: "J",
    37: "K",
    38: "L",
    50: "M",
    49: "N",
    24: "O",
    25: "P",
    16: "Q",
    19: "R",
    31: "S",
    20: "T",
    22: "U",
    47: "V",
    17: "W",
    45: "X",
    21: "Y",
    44: "Z",
}


class JointVisualizer:
    def __init__(self, trials_data):
        self.trials_data = trials_data
        self.current_trial = 0
        self.current_frame = 0
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Add buttons for navigation
        self.ax_prev_trial = plt.axes([0.1, 0.05, 0.1, 0.04])
        self.ax_next_trial = plt.axes([0.21, 0.05, 0.1, 0.04])
        self.ax_prev_frame = plt.axes([0.4, 0.05, 0.1, 0.04])
        self.ax_next_frame = plt.axes([0.51, 0.05, 0.1, 0.04])
        self.ax_first_valid = plt.axes([0.7, 0.05, 0.2, 0.04])

        self.btn_prev_trial = Button(self.ax_prev_trial, "Prev Trial")
        self.btn_next_trial = Button(self.ax_next_trial, "Next Trial")
        self.btn_prev_frame = Button(self.ax_prev_frame, "Prev Frame")
        self.btn_next_frame = Button(self.ax_next_frame, "Next Frame")
        self.btn_first_valid = Button(self.ax_first_valid, "First Valid Frame")

        self.btn_prev_trial.on_clicked(self.prev_trial)
        self.btn_next_trial.on_clicked(self.next_trial)
        self.btn_prev_frame.on_clicked(self.prev_frame)
        self.btn_next_frame.on_clicked(self.next_frame)
        self.btn_first_valid.on_clicked(self.first_valid_frame)

        # Initialize the plot
        self.update_plot()

    def prev_trial(self, event):
        if self.current_trial > 0:
            self.current_trial -= 1
            self.current_frame = 0
            self.update_plot()

    def next_trial(self, event):
        if self.current_trial < len(self.trials_data) - 1:
            self.current_trial += 1
            self.current_frame = 0
            self.update_plot()

    def prev_frame(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_plot()

    def next_frame(self, event):
        timestamp_joints = self.trials_data[self.current_trial]["timestamp_joints"]
        if self.current_frame < len(timestamp_joints) - 1:
            self.current_frame += 1
            self.update_plot()

    def first_valid_frame(self, event):
        timestamp_joints = self.trials_data[self.current_trial]["timestamp_joints"]
        for i, (_, joints) in enumerate(timestamp_joints):
            if joints is not None and len(joints) == 2:
                self.current_frame = i
                self.update_plot()
                break

    def update_plot(self):
        self.ax.clear()

        # Get current trial and frame data
        timestamp_joints = self.trials_data[self.current_trial]["timestamp_joints"]
        timestamp, joints = timestamp_joints[self.current_frame]

        # Get the typed text near the timestamp
        log_dicts = self.trials_data[self.current_trial]["logs"]
        typed_text = self.decode_keypresses_near(log_dicts, timestamp)

        # Check if joints data is valid
        if joints is not None and len(joints) == 2:
            left_hand = np.array(joints[0])
            right_hand = np.array(joints[1])

            # Invert the z-axis
            left_hand[:, 2] = -left_hand[:, 2]
            right_hand[:, 2] = -right_hand[:, 2]

            # Invert the y-axis
            left_hand[:, 1] = -left_hand[:, 1]
            right_hand[:, 1] = -right_hand[:, 1]

            # Plot left hand joints (red)
            self.ax.scatter(
                left_hand[:, 0],
                left_hand[:, 1],
                left_hand[:, 2],
                c="r",
                marker="o",
                label="Left Hand",
            )

            # Add joint numbers for left hand
            for i in range(len(left_hand)):
                self.ax.text(
                    left_hand[i, 0],
                    left_hand[i, 1],
                    left_hand[i, 2],
                    str(i),
                    color="r",
                    fontsize=8,
                )

            # Plot right hand joints (blue)
            self.ax.scatter(
                right_hand[:, 0],
                right_hand[:, 1],
                right_hand[:, 2],
                c="b",
                marker="o",
                label="Right Hand",
            )

            # Add joint numbers for right hand
            for i in range(len(right_hand)):
                self.ax.text(
                    right_hand[i, 0],
                    right_hand[i, 1],
                    right_hand[i, 2],
                    str(i),
                    color="b",
                    fontsize=8,
                )

            # Connect joints with lines to show hand structure
            self.plot_hand_connections(left_hand, "r")
            self.plot_hand_connections(right_hand, "b")

            # Set plot limits and labels
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            # Auto-adjust the axis limits
            x_data = np.concatenate((left_hand[:, 0], right_hand[:, 0]))
            y_data = np.concatenate((left_hand[:, 1], right_hand[:, 1]))
            z_data = np.concatenate((left_hand[:, 2], right_hand[:, 2]))

            x_range = np.max(x_data) - np.min(x_data)
            y_range = np.max(y_data) - np.min(y_data)
            z_range = np.max(z_data) - np.min(z_data)

            x_mid = (np.max(x_data) + np.min(x_data)) / 2
            y_mid = (np.max(y_data) + np.min(y_data)) / 2
            z_mid = (np.max(z_data) + np.min(z_data)) / 2

            max_range = max(x_range, y_range, z_range) / 2

            self.ax.set_xlim(x_mid - max_range, x_mid + max_range)
            self.ax.set_ylim(y_mid - max_range, y_mid + max_range)
            self.ax.set_zlim(z_mid - max_range, z_mid + max_range)

        # Set title with trial, frame, and timestamp information
        trial_name = list_pickle_files()[self.current_trial].replace(".pkl", "")
        self.ax.set_title(
            f"Trial: {trial_name} | Frame: {self.current_frame} | Timestamp: {timestamp}\nTyped Text: {typed_text}"
        )

        self.ax.legend()
        self.fig.canvas.draw_idle()

    def decode_keypresses_near(self, log_dicts, timestamp):
        # Find the log dicts near the timestamp
        BUFFER = 1000  # milliseconds
        downs = []
        for log_dict in log_dicts:
            if (
                log_dict["time"] > timestamp - BUFFER
                and log_dict["time"] < timestamp + BUFFER
                and log_dict["type"] == "down"
                and log_dict["keycode"] in key_dict
            ):
                downs.append(key_dict[log_dict["keycode"]])

        return " ".join(downs)

    def plot_hand_connections(self, hand_joints, color):
        # MediaPipe hand connections
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # thumb
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # index finger
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # middle finger
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # ring finger
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # pinky
        ]

        for connection in connections:
            self.ax.plot(
                [hand_joints[connection[0], 0], hand_joints[connection[1], 0]],
                [hand_joints[connection[0], 1], hand_joints[connection[1], 1]],
                [hand_joints[connection[0], 2], hand_joints[connection[1], 2]],
                color=color,
            )


def visualize_joint_data():
    # Load the trials data
    trials_data = read_pickle_files()

    if not trials_data:
        print("No trial data found. Please check the data directory.")
        return

    # Create the visualizer
    visualizer = JointVisualizer(trials_data)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    visualize_joint_data()
