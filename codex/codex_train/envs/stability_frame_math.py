"""Palm-to-actuation reference vector used by the optional stability frame."""


def palm_actuation_vector(palm_position, actuation_position, palm_rotation, stability_frame):
    vector_world = palm_position - actuation_position
    if stability_frame == "world":
        return vector_world
    if stability_frame == "hand":
        return (palm_rotation.transpose(-1, -2) @ vector_world.unsqueeze(-1)).squeeze(-1)
    raise ValueError("stability_frame must be 'world' or 'hand'")
