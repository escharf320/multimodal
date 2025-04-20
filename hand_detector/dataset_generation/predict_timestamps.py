import numpy as np
from sklearn.linear_model import LinearRegression


def predict_missing_timestamps(frame_data):
    """
    Input: frame_data (list): List of tuples (frame_number, timestamp or None).

    Returns: List of tuples (frame_number, timestamp).
    """
    # Extract known frame numbers and timestamps
    frames, known_times = zip(*[(f, t) for f, t in frame_data if t is not None])

    filled_data = []

    if frames:
        model = LinearRegression()
        model.fit(np.array(frames).reshape(-1, 1), known_times)

        for f, t in frame_data:
            if t is not None:
                filled_data.append((f, t))
            else:
                predicted = int(model.predict(np.array([[f]]))[0])
                filled_data.append((f, predicted))
    else:
        filled_data = [(f, None) for f, _ in frame_data]

    return filled_data
