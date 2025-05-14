import random

import torch
from typeformar.dataset_generation.batch_video_preparation import read_pickle_files
from typeformar.dataset_generation.loss_array.key_pressed import key_pressed_v2

SPACEBAR_KEY = 57
MIN_SEQUENCE_LENGTH = 50
WINDOW_SIZE = 15

assert WINDOW_SIZE % 2 == 1, "Window size must be odd"
assert (
    WINDOW_SIZE < MIN_SEQUENCE_LENGTH
), "Window size must be less than sequence length"

# 1. Read the trial information from the pkl files

trials_data = read_pickle_files()

# 2. Detect which frames the spacebar was pressed

spacebar_pressed_lists = [
    key_pressed_v2(
        trial_data["logs"],
        trial_data["timestamp_joints"],
        SPACEBAR_KEY,
    )
    for trial_data in trials_data
]

# 3. Extract additional features from the trials
# First, we extract the velocity of the hand
# Then, we normalize the positions of the hand on a per-hand basis


def tensorize_joints(timestamp_joints):
    """
    Tensorize the joints
    """
    assert len(timestamp_joints) == 2, "Only two hands are supported"

    left, right = timestamp_joints

    left_tensor = torch.tensor(left)
    right_tensor = torch.tensor(right)

    return left_tensor, right_tensor


def extract_velocity(left_tensor, right_tensor):
    """
    Extract the velocity of the hand from the timestamp_joints
    """
    left_velocity = torch.diff(left_tensor, dim=0)
    right_velocity = torch.diff(right_tensor, dim=0)

    return left_velocity, right_velocity


def normalize_joints(left_tensor, right_tensor):
    """
    Normalize the joints on a per-hand basis
    """
    left_normalized = (left_tensor - left_tensor.mean(dim=0)) / left_tensor.std(dim=0)
    right_normalized = (right_tensor - right_tensor.mean(dim=0)) / right_tensor.std(
        dim=0
    )

    return left_normalized, right_normalized


def extract_relative_positions_from_palms(left_tensor, right_tensor):
    """
    Extract the relative positions of the palms from the timestamp_joints
    """
    left_relative = left_tensor - left_tensor[0]
    right_relative = right_tensor - right_tensor[0]
    return left_relative, right_relative


def generate_feature_vector(timestamp_joints):
    """
    Generate a feature vector from the timestamp_joints
    """
    left_tensor, right_tensor = tensorize_joints(timestamp_joints)
    left_velocity, right_velocity = extract_velocity(left_tensor, right_tensor)
    left_normalized, right_normalized = normalize_joints(left_tensor, right_tensor)
    left_relative, right_relative = extract_relative_positions_from_palms(
        left_tensor, right_tensor
    )
    left_relative_normalized, right_relative_normalized = normalize_joints(
        left_relative, right_relative
    )
    return torch.cat(
        (
            left_tensor,
            right_tensor,
            left_normalized,
            right_normalized,
            left_velocity,
            right_velocity,
            left_relative,
            right_relative,
            left_relative_normalized,
            right_relative_normalized,
        ),
        dim=0,
    ).view(-1)


def generate_contiguous_sequences():
    """
    Generate contiguous sequences from the feature vectors
    """
    all_sequences = []

    for i, trial_datum in enumerate(trials_data[0:2]):
        last_f = -1
        current_feature_sequence = []
        current_output_sequence = []

        for f, joint_data in enumerate(trial_datum["timestamp_joints"]):
            _, joints = joint_data

            # Defined and has 2 hands
            if joints is not None and len(joints) == 2:
                feature_vector = generate_feature_vector(joints)
                output_element = 1 if spacebar_pressed_lists[i][f] else 0

                # If the current frame is the next frame, we add the feature vector
                # and the output element to the current sequence
                if f - last_f == 1:
                    current_feature_sequence.append(feature_vector)
                    current_output_sequence.append(output_element)
                # If it's not the next frame, we reset the current sequences
                else:
                    # If the current sequence is long enough, we add it to the dataset
                    if len(current_feature_sequence) >= MIN_SEQUENCE_LENGTH:
                        all_sequences.append(
                            (
                                torch.stack(current_feature_sequence),
                                torch.tensor(current_output_sequence),
                            )
                        )

                    # We reset the current sequences
                    current_feature_sequence = [feature_vector]
                    current_output_sequence = [output_element]

                # We update the last frame seen
                last_f = f

    return all_sequences


def generate_sliding_windows(all_sequences):
    """
    Generate sliding windows from the contiguous sequences.
    For each sequence, we generate a sliding window of size `WINDOW_SIZE`
    where the output is the middle element of the window.
    """
    all_windows = []

    for sequence in all_sequences:
        feature_sequence, output_sequence = sequence

        # Generate the sliding windows
        for i in range(len(feature_sequence) - WINDOW_SIZE + 1):
            window = feature_sequence[i : i + WINDOW_SIZE]
            output = torch.tensor([output_sequence[i + WINDOW_SIZE // 2]])
            all_windows.append((window, output))

    return all_windows


def prepare_dataset():
    all_sequences = generate_contiguous_sequences()
    all_windows = generate_sliding_windows(all_sequences)

    return all_windows


if __name__ == "__main__":
    # Prepare the dataset and then print a sample
    dataset = prepare_dataset()
    for feature_sequence, output_sequence in dataset:
        if output_sequence == 0:
            print(feature_sequence.shape)
            print(feature_sequence[:, 11:12])
            print(output_sequence)
            break
