import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import streamlit as st


def initialize_spotify_client(credentials_path=None):
    scope = "user-library-read playlist-read-private"
    default_credentials_path = "../creds/spotify_credentials.json"

    if credentials_path:
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"No credentials file found at {credentials_path}")
    elif os.path.exists(default_credentials_path):
        credentials_path = default_credentials_path
    else:
        credentials_path = None

    if credentials_path:
        with open(credentials_path, 'r') as file:
            creds = json.load(file)
        client_id = creds['client_id']
        client_secret = creds['client_secret']
        redirect_uri = creds['redirect_uri']
    else:
        client_id = os.getenv('SPOTIPY_CLIENT_ID')
        client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
        redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')
        if not all([client_id, client_secret, redirect_uri]):
            raise ValueError(
                "Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI in your environment variables.")

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope))
    return sp


class Playlist:
    def __init__(self, sp, playlist_url):
        self.sp = sp
        self.playlist_url = playlist_url
        self.tracks = self.extract_tracks()

    def extract_tracks(self):
        playlist_id = self.playlist_url.split('/')[-1].split('?')[0]
        tracks = []
        results = self.sp.playlist_tracks(playlist_id)

        while results:
            for item in results['items']:
                track_info = item['track']
                track_id = track_info['id']
                audio_features = self.sp.audio_features(track_id)[0]
                audio_features = {k: v for k, v in audio_features.items(
                ) if k not in ['type', 'analysis_url', 'duration_ms']}
                artists = [{'artist_name': artist['name'], 'artist_genres': self.sp.artist(
                    artist['id'])['genres']} for artist in track_info['artists']]
                new_track = Track(track_id, track_info,
                                  audio_features, artists)
                new_track.calculate_camelot_key()
                tracks.append(new_track)

            if results['next']:
                results = self.sp.next(results)
            else:
                results = None

        return tracks

    def list_songs(self):
        return [f"{track.track_name} - {', '.join(artist['artist_name'] for artist in track.artists)}" for track in self.tracks]

    def select_first_song(self, first_song=None):
        if first_song:
            first_song_lower = first_song.lower()
            first_song = next(
                (track for track in self.tracks if first_song_lower in f"{track.track_name.lower()} - {', '.join(artist['artist_name'].lower() for artist in track.artists)}"), None)
            if not first_song:
                raise ValueError(
                    "Specified song not found in the playlist. Try using just the song name or choose a random song.")
        else:
            first_song = random.choice(self.tracks)

        self.tracks.remove(first_song)
        return [first_song], self.tracks.copy()

    def select_next_song(self, setlist, remaining_tracks, bpm_range=0.05, max_songs=30):
        while remaining_tracks and len(setlist) < max_songs:
            current_song = setlist[-1]
            current_bpm = current_song.audio_features['tempo']
            current_key = current_song.camelot_key

            bpm_min, bpm_max = current_bpm * \
                (1 - bpm_range), current_bpm * (1 + bpm_range)
            filtered_songs = [song for song in remaining_tracks if bpm_min <=
                              song.audio_features['tempo'] <= bpm_max]

            for song in filtered_songs:
                song.key_compatibility_score = current_song.calculate_key_compatibility_score(
                    song)

            sorted_songs = sorted(
                filtered_songs, key=lambda x: x.key_compatibility_score, reverse=True)

            if sorted_songs:
                top_candidates = [sorted_songs[0]]
                for candidate in sorted_songs[1:]:
                    if candidate.key_compatibility_score == top_candidates[0].key_compatibility_score:
                        audio_sim, genre_sim = current_song.calculate_similarity_scores(
                            candidate)
                        similarity_score = 0.9 * audio_sim + 0.1 * genre_sim

                        if similarity_score > top_candidates[0].key_compatibility_score:
                            top_candidates = [candidate]
                        elif similarity_score == top_candidates[0].key_compatibility_score:
                            top_candidates.append(candidate)
                    else:
                        break

                next_song = random.choice(
                    top_candidates) if top_candidates else None
                if next_song:
                    remaining_tracks.remove(next_song)
                    setlist.append(next_song)
            else:
                # If no songs within the current BPM range, increase the range
                bpm_range = 0.08

        return setlist, remaining_tracks


class Track:
    def __init__(self, track_id, track_info, audio_features, artists):
        self.track_id = track_id
        self.track_name = track_info['name']
        self.audio_features = audio_features
        self.artists = artists
        self.camelot_key = None

    def calculate_camelot_key(self):
        key = self.audio_features['key']
        mode = self.audio_features['mode']
        camelot_map = {
            (0, 0): '5A', (0, 1): '8B', (1, 0): '12A', (1, 1): '3B',
            (2, 0): '7A', (2, 1): '10B', (3, 0): '2A', (3, 1): '3B',
            (4, 0): '9A', (4, 1): '12B', (5, 0): '4A', (5, 1): '7B',
            (6, 0): '11A', (6, 1): '2B', (7, 0): '6A', (7, 1): '9B',
            (8, 0): '1A', (8, 1): '4B', (9, 0): '8A', (9, 1): '11B',
            (10, 0): '3A', (10, 1): '6B', (11, 0): '10A', (11, 1): '1B'
        }

        self.camelot_key = camelot_map.get((key, mode), None)

    def calculate_similarity_scores(self, other_track):
        song1_features = [
            self.audio_features[feat] for feat in [
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
            ]
        ]
        song2_features = [
            other_track.audio_features[feat] for feat in [
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
            ]
        ]

        audio_similarity = cosine_similarity(
            [song1_features], [song2_features])[0][0]

        song1_genres = {
            genre for artist in self.artists for genre in artist['artist_genres']}
        song2_genres = {
            genre for artist in other_track.artists for genre in artist['artist_genres']}

        genre_similarity = len(song1_genres & song2_genres) / \
            len(song1_genres | song2_genres)

        return audio_similarity, genre_similarity

    def calculate_key_compatibility_score(self, other_track):
        key1 = self.camelot_key
        key2 = other_track.camelot_key

        # Same key
        if key1 == key2:
            return 1.0

        # Relative key
        if key1[-1] != key2[-1] and key1[:-1] == key2[:-1]:
            return 0.9

        # Perfect fifth up or down
        key1_num, key2_num = int(key1[:-1]), int(key2[:-1])
        if (key1_num - key2_num) % 12 == 7 and key1[-1] == key2[-1]:
            return 0.8
        if (key2_num - key1_num) % 12 == 7 and key1[-1] == key2[-1]:
            return 0.8

        # Perfect fifth up or down to the relative key
        if (key1_num - key2_num) % 12 == 7 and key1[-1] != key2[-1]:
            return 0.7
        if (key2_num - key1_num) % 12 == 7 and key1[-1] != key2[-1]:
            return 0.7

        # Perfect fourth up or down to the relative key
        if (key1_num - key2_num) % 12 == 5 and key1[-1] != key2[-1]:
            return 0.7
        if (key2_num - key1_num) % 12 == 5 and key1[-1] != key2[-1]:
            return 0.7

        # Parallel key modulation
        if key1[:-1] == key2[:-1] and key1[-1] != key2[-1]:
            return 0.6

        return 0


def write_setlist_to_file(setlist, directory="/output"):
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    base_filename = "setlist"
    extension = ".txt"
    counter = 0
    filename = os.path.join(directory, f"{base_filename}{extension}")

    while os.path.exists(filename):
        counter += 1
        filename = os.path.join(
            directory, f"{base_filename}({counter}){extension}")

    with open(filename, 'w') as file:
        for song in setlist:
            artists = ', '.join([artist['artist_name']
                                for artist in song.artists])
            file.write(
                f"[{artists}] - [{song.track_name}] ({song.audio_features['tempo']:.2f} BPM, {song.camelot_key})\n")

    st.success(f"Setlist saved to '{filename}'")
    return filename


def main():
    st.title("Spotify Setlist Generator")
    credentials_choice = st.radio(
        "How would you like to provide your Spotify credentials?",
        ('Use a credentials file', 'Set environment variables'))

    if credentials_choice == 'Use a credentials file':
        credentials_path = st.text_input(
            "Enter the path to your credentials file:", "../creds/spotify_credentials.json")
        if st.button("Initialize Spotify Client"):
            try:
                sp = initialize_spotify_client(credentials_path)
                st.session_state['sp'] = sp
                st.success("Spotify client initialized successfully.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif credentials_choice == 'Set environment variables':
        if st.button("Initialize Spotify Client"):
            try:
                sp = initialize_spotify_client()
                st.session_state['sp'] = sp
                st.success("Spotify client initialized successfully.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if 'sp' in st.session_state:
        playlist_url = st.text_input("Enter the URL of the Spotify playlist:")
        if playlist_url:
            playlist = Playlist(st.session_state['sp'], playlist_url)
            song_choice_method = st.radio(
                "Choose the starting song method:",
                ('Specific song', 'Random song'))

            if song_choice_method == 'Specific song':
                song_choice = st.selectbox(
                    "Select a song to start with:",
                    playlist.list_songs())
                if st.button("Generate Setlist"):
                    setlist, remaining_tracks = playlist.select_first_song(
                        song_choice)
                    setlist, remaining_tracks = playlist.select_next_song(
                        setlist, remaining_tracks)
                    st.write("Final setlist:")
                    for song in setlist:
                        st.write(
                            f"[{', '.join(artist['artist_name'] for artist in song.artists)}] - [{song.track_name}] ({song.audio_features['tempo']:.2f} BPM, {song.camelot_key})")
            elif song_choice_method == 'Random song':
                if st.button("Generate Setlist"):
                    setlist, remaining_tracks = playlist.select_first_song()
                    setlist, remaining_tracks = playlist.select_next_song(
                        setlist, remaining_tracks)
                    st.write("Final setlist:")
                    for song in setlist:
                        st.write(
                            f"[{', '.join(artist['artist_name'] for artist in song.artists)}] - [{song.track_name}] ({song.audio_features['tempo']:.2f} BPM, {song.camelot_key})")

            if st.button("Save Setlist to File"):
                filename = write_setlist_to_file(setlist)
                st.write(f"Setlist saved to '{filename}'")


if __name__ == "__main__":
    main()
