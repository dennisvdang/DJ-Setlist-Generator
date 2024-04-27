import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def initialize_spotify_client(credentials_path=None):
    scope = "user-library-read playlist-read-private"
    default_credentials_path = "../creds/spotify_credentials.json"

    # Check if a specific path is provided or if the default path exists
    if credentials_path:
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"No credentials file found at {credentials_path}")
    elif os.path.exists(default_credentials_path):
        credentials_path = default_credentials_path
    else:
        # If no file is found at the default or specified path, prompt for a path
        credentials_path = input(
            "Enter the path to your Spotify credentials file: ")
        if not os.path.exists(credentials_path):
            print(
                "No file found at the specified path. Falling back to environment variables.")
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

        # Loop through each page of tracks and append to the list
        while results:
            for item in tqdm(results['items'], desc="Extracting tracks"):
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

            # Check if more tracks are available and fetch the next page
            if results['next']:
                results = self.sp.next(results)
            else:
                results = None

        return tracks

    def list_songs(self):
        return [f"{track.track_name} - {', '.join(artist['artist_name'] for artist in track.artists)}" for track in self.tracks]

    def select_first_song(self, first_song=None):
        try:
            if first_song:
                first_song_lower = first_song.lower()
                # Find the track by checking if the input is a substring of the full song name and artist string
                first_song = next(
                    (track for track in self.tracks if first_song_lower in f"{track.track_name.lower()} - {', '.join(artist['artist_name'].lower() for artist in track.artists)}"), None)
                if not first_song:
                    raise ValueError(
                        "Specified song not found in the playlist. Try using just the song name or choose a random song.")
            else:
                first_song = random.choice(self.tracks)

            self.tracks.remove(first_song)
            return [first_song], self.tracks.copy()
        except Exception as e:
            print(f"An error occurred: {e}")
            response = input(
                "Enter a song name to retry or type 'random' to select a random song: ").lower()
            if response == 'random':
                return self.select_first_song()
            else:
                # Treat any other input as a retry attempt
                return self.select_first_song(response)

    def select_next_song(self, setlist, remaining_tracks, bpm_range=0.05, max_songs=30):
        while remaining_tracks and len(setlist) < max_songs:
            current_song = setlist[-1]
            current_bpm = current_song.audio_features['tempo']
            current_key = current_song.camelot_key

            bpm_min, bpm_max = current_bpm * \
                (1 - bpm_range), current_bpm * (1 + bpm_range)
            filtered_songs = [song for song in remaining_tracks
                              if bpm_min <= song.audio_features['tempo'] <= bpm_max]

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

    print(f"Setlist saved to '{filename}'")
    return filename


def main():
    while True:  # Loop to allow generating multiple setlists
        print("How would you like to provide your Spotify credentials?")
        print("1. Use a credentials file")
        print("2. Set environment variables")
        choice = input("Enter 1 or 2: ")

        try:
            if choice == "1":
                default_credentials_path = "../creds/spotify_credentials.json"
                if os.path.exists(default_credentials_path):
                    sp = initialize_spotify_client(default_credentials_path)
                else:
                    print(
                        f"No credentials file found at the default path: {default_credentials_path}")
                    credentials_path = input(
                        "Enter the path to your credentials file: ")
                    sp = initialize_spotify_client(credentials_path)
            elif choice == "2":
                print("Please set the following environment variables:")
                print("SPOTIPY_CLIENT_ID")
                print("SPOTIPY_CLIENT_SECRET")
                print("SPOTIPY_REDIRECT_URI")
                input("Press Enter to continue once you've set the variables...")
                sp = initialize_spotify_client()
            else:
                print("Invalid choice. Exiting.")
                return
        except Exception as e:
            print(f"An error occurred while initializing Spotify client: {e}")
            return

        playlist_url = input(
            "Enter the URL of the Spotify playlist you want to generate a setlist from: ")
        playlist = Playlist(sp, playlist_url)

        print("Would you like to:")
        print("1. Choose a specific song")
        print("2. Start with a random song")
        start_choice = input("Enter 1 or 2: ")

        if start_choice == "1":
            while True:
                print("Here are the songs available:")
                for song in playlist.list_songs():
                    print(song)
                song_choice = input(
                    "Enter the song name or 'song name - artist' to start with, or type 'exit' to quit: ")
                if song_choice.lower() == 'exit':
                    print("Exiting program.")
                    return
                setlist, remaining_tracks = playlist.select_first_song(
                    song_choice)
                break
        elif start_choice == "2":
            setlist, remaining_tracks = playlist.select_first_song()

        setlist, remaining_tracks = playlist.select_next_song(
            setlist, remaining_tracks)

        print("Final setlist:")
        for song in setlist:
            artists = ', '.join([artist['artist_name']
                                for artist in song.artists])
            print(
                f"[{artists}] - [{song.track_name}] ({song.audio_features['tempo']:.2f} BPM, {song.camelot_key})")

        # Ask user if they want to save the setlist as a text file
        save_response = input(
            "Would you like to save this setlist as a text file? [Y/N]: ").lower()
        if save_response == 'y':
            filename = write_setlist_to_file(setlist)
            print(f"Setlist saved to '{filename}'")

        # Ask user for next action
        print("What would you like to do next?")
        print("1. Generate another setlist from the same playlist")
        print("2. Use a different playlist")
        print("3. Exit")
        next_action = input("Enter your choice (1, 2, or 3): ")

        if next_action == "1":
            continue  # Reuse the same playlist object
        elif next_action == "2":
            playlist = None  # Reset playlist to allow entering a new URL
            continue
        elif next_action == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Exiting.")
            break


if __name__ == "__main__":
    main()
