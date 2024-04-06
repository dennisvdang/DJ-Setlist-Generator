import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import tqdm
import time
import json


def get_playlist_info(sp, playlist_url):
    """
    Retrieve tracks, audio features, and metadata from a Spotify playlist URL.

    Args:
        sp (spotipy.Spotify): A Spotify client object.
        playlist_url (str): The URL of the Spotify playlist.

    Returns:
        dict: A nested dictionary containing the following keys:
            - 'playlist_name' (str): The name of the playlist.
            - 'tracks' (list): A list of dictionaries containing the following keys for each track:
                - 'track_name' (str): The name of the track.
                - 'artists' (list): A list of dictionaries containing the following keys for each artist:
                    - 'artist_name' (str): The name of the artist.
                    - 'artist_genres' (list): A list of genres associated with the artist.
                - 'audio_features' (dict): A dictionary containing the audio features for the track, excluding the keys
                  'type', 'analysis_url', and 'duration_ms'.
    """
    # Extract playlist ID from URL
    playlist_id = playlist_url.split('/')[-1].split('?')[0]

    # Get playlist metadata
    playlist_metadata = sp.playlist(playlist_id)
    playlist_name = playlist_metadata['name']

    # Get playlist tracks
    playlist_tracks = sp.playlist_tracks(playlist_id)['items']
    tracks = []

    for track in playlist_tracks:
        track_name = track['track']['name']
        artists = []
        for artist in track['track']['artists']:
            artist_name = artist['name']
            artist_id = artist['id']
            artist_genres = sp.artist(artist_id)['genres']
            artists.append({'artist_name': artist_name,
                           'artist_genres': artist_genres})
        track_id = track['track']['id']

        # Get audio features for the track
        audio_features_dict = sp.audio_features(track_id)[0]
        audio_features_dict = {k: v for k, v in audio_features_dict.items(
        ) if k not in ['type', 'analysis_url', 'duration_ms']}

        tracks.append({
            'track_name': track_name,
            'artists': artists,
            'audio_features': audio_features_dict
        })

    return {
        'playlist_name': playlist_name,
        'tracks': tracks
    }


def select_first_song(playlist_data, randomize=True, track_name=None):
    """
    Selects the first song from the provided playlist data.

    Args:
        playlist_data (dict): A dictionary containing the playlist name and tracks.
        randomize (bool, optional): If True, randomly selects the first song. If False, requires a track_name.
        track_name (str, optional): The name of the track to select as the first song. Required if randomize is False.

    Returns:
        dict: A dictionary representing the first song in the setlist.

    Raises:
        ValueError: If randomize is False and track_name is not provided or not found in the playlist.
    """
    tracks = playlist_data['tracks']

    if randomize:
        first_song = random.choice(tracks)
    else:
        if not track_name:
            raise ValueError(
                "If randomize is False, track_name must be provided.")

        for track in tracks:
            if track['track_name'] == track_name:
                first_song = track
                break
        else:
            raise ValueError(
                f"Track '{track_name}' not found in the playlist.")

    return first_song


def get_camelot_key(key, mode):
    """
    Convert Spotify key and mode values to Camelot notation.

    Args:
        key (int): The key value from Spotify (0-11).
        mode (int): The mode value from Spotify (0 for minor, 1 for major).

    Returns:
        str: The key in Camelot notation (e.g., 'C', 'C#', 'Dm').
    """
    camelot_keys = ['C', 'C#', 'D', 'D#', 'E',
                    'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    camelot_key = camelot_keys[key]
    if mode == 0:
        camelot_key += 'm'
    return camelot_key


def get_next_song(playlist, current_song, bpm_range=0.05, key_priority=['same', 'adjacent', 'fifth']):
    """Get the next song in the setlist based on BPM and key."""
    current_bpm = current_song['tempo']
    current_key = get_camelot_key(current_song['key'], current_song['mode'])

    # Filter songs within BPM range
    bpm_min = current_bpm * (1 - bpm_range)
    bpm_max = current_bpm * (1 + bpm_range)
    candidates = [song for song in playlist if bpm_min <=
                  song['tempo'] <= bpm_max]

    # Sort candidates by key priority
    for priority in key_priority:
        if priority == 'same':
            key_candidates = [song for song in candidates if get_camelot_key(
                song['key'], song['mode']) == current_key]
            if key_candidates:
                return key_candidates[0]
        elif priority == 'adjacent':
            adjacent_keys = [
                current_key[0] + str(int(current_key[1:]) + 1), current_key[0] + str(int(current_key[1:]) - 1)]
            key_candidates = [song for song in candidates if get_camelot_key(
                song['key'], song['mode']) in adjacent_keys]
            if key_candidates:
                return key_candidates[0]
        elif priority == 'fifth':
            fifth_keys = [current_key[0] + str((int(current_key[1:]) + 7) %
                                               12 + 1), current_key[0] + str((int(current_key[1:]) - 5) % 12 + 1)]
            key_candidates = [song for song in candidates if get_camelot_key(
                song['key'], song['mode']) in fifth_keys]
            if key_candidates:
                return key_candidates[0]

    # If no match found, return the first candidate
    return candidates[0] if candidates else None


# Load Spotify credentials
credentials_path = r'../creds/spotify_credentials.json'
with open(credentials_path, 'r') as file:
    creds = json.load(file)

# Initialize Spotify client
auth_manager = SpotifyClientCredentials(
    client_id=creds['client_id'], client_secret=creds['client_secret'])
sp = spotipy.Spotify(auth_manager=auth_manager)
