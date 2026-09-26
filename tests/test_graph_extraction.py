"""Smoke test for the deterministic graph extraction in src/utils.py.

Builds a synthetic two-room plan: a square exterior wall split down the
middle by an interior wall with a door gap. The extractor should find
exactly two rooms and a single door edge between them, because the door
adjacency takes precedence over the wall adjacency for the same pair.
"""

import numpy as np

from src.utils import (
    conn_components,
    extract_all_adjacencies,
    rooms_with_bounds,
    stack_normalize,
)

SIZE = 32


def two_room_plan():
    inside_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    inside_mask[1:31, 1:31] = 1

    # Exterior wall ring
    boundary_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    boundary_mask[1, 1:31] = 1
    boundary_mask[30, 1:31] = 1
    boundary_mask[1:31, 1] = 1
    boundary_mask[1:31, 30] = 1

    # Interior wall down column 15 with a door in rows 12-17
    room_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    room_mask[2:30, 15] = 1
    room_mask[12:18, 15] = 0

    door_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    door_mask[12:18, 15] = 1

    return inside_mask, boundary_mask, room_mask, door_mask


def test_two_rooms_joined_by_door():
    inside_mask, boundary_mask, room_mask, door_mask = two_room_plan()

    stacked_layers = stack_normalize(boundary_mask, room_mask, door_mask)
    num_labels, labels, _, _ = conn_components(
        inside_mask, boundary_mask, room_mask, door_mask
    )
    rwb = rooms_with_bounds(stacked_layers, labels)
    edges = extract_all_adjacencies(rwb)

    # Background plus two rooms
    assert num_labels == 3

    # One edge, room 5 <-> room 6, typed as a door (adj_type 1)
    assert len(edges) == 1
    edge = edges.iloc[0]
    assert (edge["n1"], edge["n2"], edge["adj_type"]) == (5, 6, 1)
    # One horizontal scan line per door row
    assert edge["edge_strength"] == 6
