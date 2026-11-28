"""hidden_roster (28 Nov 2026): an empty roster read for a session the listing
counts bookings for is a hidden roster, not a session everyone left."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from capture_roster_sightings import hidden_roster


def test_empty_read_with_bookings_listed_is_hidden():
    assert hidden_roster([], {"capacity": 20, "spots_left": 6})
    assert hidden_roster([], {"capacity": 4, "spots_left": 0})


def test_empty_read_of_an_empty_session_is_real():
    assert not hidden_roster([], {"capacity": 20, "spots_left": 20})


def test_a_roster_with_players_is_never_hidden():
    assert not hidden_roster([{"id": 1}], {"capacity": 20, "spots_left": 6})


def test_unknown_counts_are_not_hidden():
    assert not hidden_roster([], {"capacity": None, "spots_left": 3})
    assert not hidden_roster([], {"capacity": 8, "spots_left": None})
