import torch
from typeformar.dataset_generation.batch_video_preparation import read_pickle_files
from typeformar.dataset_generation.loss_array.key_pressed import key_pressed_v2

SPACEBAR_KEY = 57
MIN_SEQUENCE_LENGTH = 10

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


def generate_feature_vector(timestamp_joints):
    """
    Generate a feature vector from the timestamp_joints
    """
    left_tensor, right_tensor = tensorize_joints(timestamp_joints)
    left_velocity, right_velocity = extract_velocity(left_tensor, right_tensor)
    left_normalized, right_normalized = normalize_joints(left_tensor, right_tensor)
    return torch.cat(
        (
            left_tensor,
            right_tensor,
            left_normalized,
            right_normalized,
            left_velocity,
            right_velocity,
        ),
        dim=0,
    ).view(-1)


def prepare_dataset():
    all_sequences = []

    for trial_datum in trials_data:
        last_f = -1
        current_feature_sequence = []
        current_output_sequence = []

        for f, joint_data in enumerate(trial_datum["timestamp_joints"]):
            _, joints = joint_data

            # Defined and has 2 hands
            if joints is not None and len(joints) == 2:
                feature_vector = generate_feature_vector(joints)
                output_element = 1 if spacebar_pressed_lists[0][f] else 0

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

    # Then, go through the all sequences and cut them into sequences of the same
    # exact format, but by searching for the times where there are 1's consecutively
    # and then cutting the sequences at those times with padding of size 5

    new_all_sequences = []

    for feature_sequence, output_sequence in all_sequences:
        # Find the times where there are 1's consecutively
        # and then cut the sequences at those times with padding of size 5

        # Find the times where there are 1's consecutively
        BUFFER = 5
        for i in range(len(output_sequence)):
            if output_sequence[i] == 1 and output_sequence[i + 1] != 1:
                # Find the start of the sequence
                start = i
                while start > 0 and output_sequence[start - 1] == 1:
                    start -= 1
                new_all_sequences.append(
                    (
                        feature_sequence[start - BUFFER : i + BUFFER + 1],
                        output_sequence[start - BUFFER : i + BUFFER + 1],
                    )
                )

    return new_all_sequences


if __name__ == "__main__":
    # Prepare the dataset and then print a sample
    dataset = prepare_dataset()
    for feature_sequence, output_sequence in dataset:
        print(feature_sequence)
        print(output_sequence)
