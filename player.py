# # import argparse
# # from tabulate import tabulate
# # import json
# # import random
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity

# # def main(target_mood_str, top_n, logging = False):
# #     # Load the tags data
# #     with open('tags.json') as f:
# #         tags_data = json.load(f)
# #         print(f"Loaded {len(tags_data)} tag data items. (dict)")

# #     # --- Model Loading ---
# #     model = SentenceTransformer('sentence-transformers/sentence-t5-base', device='mps')
# #     print("Model loaded.")

# #     # --- Part 1: Prepare Strings and Generate Embeddings ---
# #     strings_to_encode = []
# #     original_indices = {}  # Store original index for later retrieval of filename

# #     if logging:
# #         print("\nProcessing and shuffling mood tags...")
# #     current_index = 0
# #     for filename, mood_list in tags_data.items():
# #         # Copy the list to avoid modifying the original dict entry if script is rerun
# #         shuffled_moods = mood_list[:]
# #         random.shuffle(shuffled_moods)
# #         combined_mood_string = ", ".join(shuffled_moods)

# #         strings_to_encode.append(combined_mood_string)
# #         original_indices[current_index] = filename  # Map current index to filename
# #         current_index += 1

# #     # Add the target string to the list for encoding
# #     strings_to_encode.append(target_mood_str)
# #     target_label = f"Target: {target_mood_str}"

# #     # Keep track of the index of the target string
# #     target_index = len(strings_to_encode) - 1
# #     original_indices[target_index] = target_label  # Also add target label here for completeness

# #     if logging:
# #         print(f"\nStrings prepared for encoding ({len(strings_to_encode)} total):")
# #         for i, s in enumerate(strings_to_encode):
# #             print(f"  {i}: '{s}' (Label: '{original_indices[i]}')")

# #     if logging:
# #         print(f"\nEncoding {len(strings_to_encode)} strings...")

# #     embeddings = model.encode(["music that is " + x for x in strings_to_encode])

# #     # --- Part 4: Calculate and Print Similarities ---

# #     if logging:
# #         print("\nCalculating similarities with target:", f"'{target_mood_str}'")

# #     # Extract the target embedding
# #     target_embedding = embeddings[target_index]

# #     # Prepare lists for non-target embeddings and their labels
# #     other_embeddings = []
# #     other_labels = []
# #     for i in range(len(embeddings)):
# #         if i != target_index:
# #             other_embeddings.append(embeddings[i])
# #             other_labels.append(original_indices[i])  # Get the original filename

# #     # Ensure there are other embeddings to compare against
# #     if not other_labels:
# #         print("No other files found to compare the target against.")
# #     else:
# #         # Calculate cosine similarities between the target and all others
# #         # Reshape target_embedding to (1, N) and other_embeddings should be (M, N)
# #         similarities = cosine_similarity([target_embedding], other_embeddings)[0]  # Get the first row

# #         # Pair labels with their similarity scores
# #         results = list(zip(other_labels, similarities))

# #         # Sort by similarity score in descending order
# #         results_sorted = sorted(results, key=lambda item: item[1], reverse=True)

# #         # Print the sorted list if logging
# #         headers = ["#", "File", "Similarity"]
# #         table_data = []
# #         for rank, (label, score) in enumerate(results_sorted[:top_n]):
# #             table_data.append([rank+1, label, f"{score:.4f}"])

# #         print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

# #         return results_sorted[:top_n]

# #     if logging:
# #         print("\nDone.")

# # if __name__ == "__main__":
# #     # Set up command-line argument parsing
# #     parser = argparse.ArgumentParser(description='Find music files with a specified mood.')
# #     parser.add_argument('-mood', '--mood', type=str, required=True, help='Comma-separated list of mood tags, with spaces within tags allowed using quotes.')
# #     parser.add_argument('-top', '--top', type=int, default=10, help='Number of top results to display.')

# #     # Parse the arguments
# #     args = parser.parse_args()

# #     # Call the main function with the provided arguments
# #     main(args.mood, args.top)

# import argparse
# import json
# import random
# import os
# import time
# import math
# import sys
# import pygame
# from tabulate import tabulate
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment
# from pydub.exceptions import CouldntDecodeError

# # --- Configuration ---
# MP3_FOLDER = "static/mp3"  # Folder containing your mp3 files (must match tags.json location assumption)
# TAGS_FILE = 'tags.json' # Location of your tags data
# SAMPLE_MP3_FILENAME = "Deep Stone Crypt Theme.mp3"  # Reference volume MP3 (ensure this exists in MP3_FOLDER)
# MODEL_NAME = 'sentence-transformers/sentence-t5-base' # Or another suitable model
# # Set to 'cuda', 'mps' (for Apple Silicon), or 'cpu'
# DEVICE = 'mps' if sys.platform == "darwin" else ('cuda' if torch.cuda.is_available() else 'cpu')
# # ---------------------


# # --- Helper Functions from Script 2 (Volume Normalization) ---

# def get_audio_peak_level(filepath):
#     """Calculates the peak audio level (max_dBFS) of an audio file."""
#     try:
#         audio = AudioSegment.from_mp3(filepath + ".mp3")
#         if audio.duration_seconds > 0:
#             peak_dbfs = audio.max_dBFS
#             if peak_dbfs == -math.inf:
#                  # print(f"Warning: File {os.path.basename(filepath)} appears to be silent. Assigning very low peak level.")
#                  return -100.0
#             return peak_dbfs
#         else:
#             # print(f"Warning: File {os.path.basename(filepath)} has zero duration. Assigning very low peak level.")
#             return -100.0
#     except CouldntDecodeError:
#         print(f"Error: Could not decode file: {os.path.basename(filepath)}. Skipping analysis.")
#         return None
#     except FileNotFoundError:
#         print(f"Error: File not found during analysis: {filepath}. Skipping.")
#         return None
#     except Exception as e:
#         print(f"Error analyzing peak level for {os.path.basename(filepath)}: {e}. Skipping analysis.")
#         return None

# def calculate_volume_scale(target_peak_dbfs, current_peak_dbfs):
#     """
#     Calculates the pygame volume scale factor (0.0 to 1.0) needed
#     to adjust current_peak_dbfs towards target_peak_dbfs.
#     """
#     if target_peak_dbfs is None or current_peak_dbfs is None:
#         return 0.5 # Default volume if analysis failed

#     if target_peak_dbfs <= -100.0 or current_peak_dbfs <= -100.0:
#          # print("Warning: Encountered extremely low peak levels, using default scale 0.5")
#          return 0.5

#     db_difference = target_peak_dbfs - current_peak_dbfs
#     # Use dB difference for peak normalization (gain = 10^(dB/20))
#     # Clamping to 1.0 ensures we only attenuate peaks louder than the target.
#     scale_factor = 10**(db_difference / 20.0)
#     scaled_volume = max(0.0, min(1.0, scale_factor))
#     return scaled_volume

# # --- Main Combined Logic ---

# def main(target_mood_str, top_n, logging=False):
#     # === Part 1: Find Top N Files Matching Mood (Adapted from Script 1) ===

#     print("--- Mood Matching Phase ---")
#     # --- Load Tags Data ---
#     tags_filepath = TAGS_FILE # Assume tags.json is inside mp3 folder or adjust path
#     if not os.path.exists(tags_filepath):
#          print(f"Error: Tags file not found at {tags_filepath}")
#          sys.exit(1)
#     try:
#         with open(tags_filepath) as f:
#             tags_data = json.load(f)
#             if not tags_data:
#                 print("Error: Tags data file is empty.")
#                 sys.exit(1)
#             print(f"Loaded {len(tags_data)} tag data items from {tags_filepath}.")
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {tags_filepath}.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"Error loading tags file: {e}")
#         sys.exit(1)


#     # --- Load Model ---
#     print(f"Loading sentence transformer model '{MODEL_NAME}' onto device '{DEVICE}'...")
#     try:
#         # Explicitly check for torch if needed for device selection clarification
#         # import torch # Uncomment if using the torch check below
#         # DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
#         model = SentenceTransformer(MODEL_NAME, device=DEVICE)
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading Sentence Transformer model: {e}")
#         print("Please ensure sentence-transformers and its dependencies (like PyTorch) are installed.")
#         sys.exit(1)

#     # --- Prepare Strings and Generate Embeddings ---
#     strings_to_encode = []
#     original_indices = {}  # Map embedding index to original filename (basename)
#     filenames_in_tags = list(tags_data.keys()) # Keep track of filenames we process

#     if logging:
#         print("\nProcessing mood tags...")

#     current_index = 0
#     for filename in filenames_in_tags:
#         mood_list = tags_data.get(filename, [])
#         if not mood_list:
#             if logging:
#                 print(f"  Skipping '{filename}': No mood tags found.")
#             continue # Skip files with no tags

#         # Shuffle tags for robustness (optional, kept from original)
#         shuffled_moods = mood_list[:]
#         random.shuffle(shuffled_moods)
#         combined_mood_string = ", ".join(shuffled_moods)

#         strings_to_encode.append(combined_mood_string)
#         original_indices[current_index] = filename  # Map current index to filename
#         current_index += 1

#     if not strings_to_encode:
#         print("Error: No files with tags found to process.")
#         sys.exit(1)

#     # Add the target string
#     strings_to_encode.append(target_mood_str)
#     target_label = f"Target: {target_mood_str}"
#     target_index = len(strings_to_encode) - 1
#     original_indices[target_index] = target_label

#     if logging:
#         print(f"\nEncoding {len(strings_to_encode)} strings (target + {len(strings_to_encode)-1} files)...")

#     # Prepend "music that is " for better context (kept from original)
#     embeddings = model.encode(["music that is " + x for x in strings_to_encode], show_progress_bar=logging)
#     print("Embeddings generated.")

#     # --- Calculate Similarities ---
#     if logging:
#         print("\nCalculating similarities with target:", f"'{target_mood_str}'")

#     target_embedding = embeddings[target_index]
#     other_embeddings = []
#     other_labels = [] # These will be the filenames
#     for i in range(len(embeddings)):
#         if i != target_index:
#             other_embeddings.append(embeddings[i])
#             other_labels.append(original_indices[i]) # Get the original filename

#     if not other_labels:
#         print("Error: No file embeddings generated to compare against target.")
#         sys.exit(1)

#     similarities = cosine_similarity([target_embedding], other_embeddings)[0]
#     results = list(zip(other_labels, similarities))
#     results_sorted = sorted(results, key=lambda item: item[1], reverse=True)

#     # --- Get Top N Filenames ---
#     top_results = results_sorted[:top_n]

#     if not top_results:
#         print(f"No files found matching the mood '{target_mood_str}'.")
#         sys.exit(1)

#     # Print the mood matching results table
#     print("\n--- Top Mood Matches ---")
#     headers = ["#", "File", "Similarity"]
#     table_data = []
#     top_filenames = [] # Store just the filenames for playback
#     for rank, (label, score) in enumerate(top_results):
#         table_data.append([rank + 1, label, f"{score:.4f}"])
#         top_filenames.append(label) # Add filename to list

#     print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

#     # Construct full paths for the selected files
#     top_files_full_paths = [os.path.join(MP3_FOLDER, fname) for fname in top_filenames]

#     # === Part 2: Volume Normalization and Playback (Adapted from Script 2) ===

#     print("\n--- Volume Normalization and Playback Phase ---")
#     # --- Basic Setup and File Checks ---
#     if not os.path.isdir(MP3_FOLDER):
#         print(f"Error: MP3 folder not found: {MP3_FOLDER}")
#         sys.exit(1)

#     sample_filepath = os.path.join(MP3_FOLDER, SAMPLE_MP3_FILENAME)
#     if not os.path.isfile(sample_filepath):
#         print(f"Error: Sample MP3 for volume reference not found: {sample_filepath}")
#         sys.exit(1)

#     # --- Initialize Pygame ---
#     print("Initializing Pygame Mixer...")
#     try:
#         pygame.mixer.pre_init(44100, -16, 2, 2048)
#         pygame.init()
#         pygame.mixer.init()
#         print("Pygame initialized successfully.")
#     except pygame.error as e:
#         print(f"Error initializing Pygame: {e}")
#         print("Ensure audio drivers are installed and configured.")
#         sys.exit(1)

#     # --- Analyze Required Audio Files ---
#     # We need to analyze the top N files selected *and* the sample file
#     files_to_analyze_set = set(top_files_full_paths)
#     files_to_analyze_set.add(sample_filepath) # Ensure sample is included
#     files_to_analyze = list(files_to_analyze_set)

#     print(f"\nAnalyzing {len(files_to_analyze)} audio files for peak levels...")
#     audio_data = {} # Store analysis results: {filepath: {'peak_dbfs': value, 'scale': value}}

#     for filepath_pre in files_to_analyze:
#         filepath = filepath_pre + ".mp3"
#         if not os.path.exists(filepath):
#             print(f"Warning: File path from tags/selection does not exist: {filepath}. Skipping analysis.")
#             continue
#         filename = os.path.basename(filepath)
#         if logging:
#             print(f"  Analyzing: {filename}")
#         peak_dbfs = get_audio_peak_level(filepath)
#         if peak_dbfs is not None:
#             audio_data[filepath] = {'peak_dbfs': peak_dbfs, 'scale': 1.0} # Default scale
#         # Error message printed inside get_audio_peak_level

#     if not audio_data:
#         print("Error: No audio files could be successfully analyzed.")
#         pygame.quit()
#         sys.exit(1)

#     # --- Calculate Target Peak Level and Scaling Factors ---
#     if sample_filepath not in audio_data:
#          print(f"\nError: Sample file '{SAMPLE_MP3_FILENAME}' was found but could not be analyzed.")
#          pygame.quit()
#          sys.exit(1)

#     target_peak_dbfs = audio_data[sample_filepath]['peak_dbfs']
#     print(f"\nReference Peak Level (from {SAMPLE_MP3_FILENAME}): {target_peak_dbfs:.2f} dBFS")

#     if target_peak_dbfs is None or target_peak_dbfs <= -100.0:
#         print("Error: Could not determine a valid peak level for the sample file. Cannot scale.")
#         pygame.quit()
#         sys.exit(1)

#     print("\nCalculating volume scales for the top files:")
#     # Calculate scales ONLY for the files we intend to play (the top N)
#     for filepath in top_files_full_paths:
#         if filepath in audio_data: # Check if analysis was successful
#             data = audio_data[filepath]
#             scale = calculate_volume_scale(target_peak_dbfs, data['peak_dbfs'])
#             audio_data[filepath]['scale'] = scale # Update scale in our data dict
#             if logging:
#                  print(f"  - {os.path.basename(filepath)}:")
#                  print(f"      Peak Level: {data['peak_dbfs']:.2f} dBFS -> Scale: {scale:.3f}")
#         else:
#              print(f"  - Skipping scale calculation for {os.path.basename(filepath)} (analysis failed or file missing). Will use default scale 0.5.")
#              # Add placeholder data if missing, so playback doesn't crash
#              if filepath not in audio_data:
#                  audio_data[filepath] = {'peak_dbfs': None, 'scale': 0.5}


#     # --- Prepare Playback List (the top N files) ---
#     playback_list = top_files_full_paths[:] # Copy the list
#     # --- Playback Loop ---
#     print(f"\n--- Starting Looping Playback of Top {len(playback_list)} Mood Matches (Endless Loop) ---")
#     try:
#         while True: # Loop forever
#             random.shuffle(playback_list) # Shuffle the order for each loop iteration

#             for filepath_pre in playback_list:
#                 if filepath_pre not in audio_data:
#                     print(f"\nSkipping {os.path.basename(filepath_pre)}: Missing analysis data.")
#                     continue

#                 filepath = filepath_pre + ".mp3"
#                 filename = os.path.basename(filepath)
#                 # Use calculated scale, default to 0.5 if analysis failed but we added placeholder
#                 volume_scale = audio_data[filepath].get('scale', 0.8)
#                 original_peak_dbfs = audio_data[filepath].get('peak_dbfs', 'N/A')
#                 peak_str = f"{original_peak_dbfs:.2f}" if isinstance(original_peak_dbfs, float) else original_peak_dbfs

#                 print(f"\n▶️ Playing: {filename}")
#                 print(f"  Original Peak: {peak_str} dBFS | Target Scale: {volume_scale:.3f}")

#                 try:
#                     sound = pygame.mixer.Sound(filepath)
#                     sound.set_volume(volume_scale)
#                     sound.play()

#                     # Wait for the sound to finish playing + handle events
#                     while pygame.mixer.get_busy():
#                         for event in pygame.event.get():
#                             if event.type == pygame.QUIT:
#                                 print("\nQuit event detected. Stopping playback.")
#                                 pygame.mixer.stop()
#                                 raise SystemExit
#                         pygame.time.Clock().tick(10) # Prevent high CPU usage

#                 except pygame.error as e:
#                     print(f"  Error playing {filename}: {e}")
#                 except Exception as e:
#                      print(f"  An unexpected error occurred during playback of {filename}: {e}")

#     except SystemExit:
#         print("Playback stopped by user.")
#     except KeyboardInterrupt:
#         print("\nPlayback interrupted by user (Ctrl+C).")
#     finally:
#         print("\nPlayback finished or stopped. Cleaning up...")
#         pygame.mixer.quit()
#         pygame.quit()
#         print("Done.")


# # --- Script Entry Point ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Find and play music files matching a specified mood with volume normalization.')
#     parser.add_argument('-mood', '--mood', type=str, required=True,
#                         help='Target mood description (e.g., "epic space battle", "calm ambient study").')
#     parser.add_argument('-top', '--top', type=int, default=5,
#                         help='Number of top matching files to find and play.')
#     parser.add_argument('--log', action='store_true',
#                         help='Enable detailed logging output.')
#     # Optional: Add arguments for MP3_FOLDER, SAMPLE_MP3_FILENAME, TAGS_FILE if needed
#     parser.add_argument('--folder', type=str, default="./static/mp3/", help='Path to the MP3 folder.')
#     parser.add_argument('--sample', type=str, default="Deep Stone Crypt Theme.mp3", help='Filename of the reference volume MP3.')

#     args = parser.parse_args()

#     # --- Dependency Check (Optional but recommended) ---
#     try:
#         import torch # Needed for sentence-transformer device check
#         import sentence_transformers
#         import sklearn
#         import pydub
#         import pygame
#         from tabulate import tabulate
#     except ImportError as e:
#         print(f"Error: Missing required Python package: {e.name}")
#         print("Please install required packages, e.g., pip install sentence-transformers scikit-learn pydub pygame tabulate torch")
#         # Note: pydub might require ffmpeg/libav installed on your system.
#         # Note: pygame might have OS-specific dependencies.
#         sys.exit(1)

#     # Call the main function
#     main(args.mood, args.top, logging=args.log)


import argparse
import json
import random
import os
import time
import math
import sys
import pygame
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import torch # Import torch to check availability for device setting

# --- Configuration ---
# MP3_FOLDER should be relative to where you run the script
MP3_FOLDER = "static/mp3"
# TAGS_FILE is also relative to where you run the script
TAGS_FILE = 'tags.json' # Assumes tags.json is in the same directory you run the script from
SAMPLE_MP3_FILENAME = "Deep Stone Crypt Theme.mp3"  # Reference MP3 (must include extension and exist in MP3_FOLDER)
MODEL_NAME = 'sentence-transformers/sentence-t5-base' # Or another suitable model

# --- Dynamically Set Device ---
if sys.platform == "darwin":
    # Check for MPS availability on macOS
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
# ---------------------


# --- Helper Functions (Volume Normalization) ---

def get_audio_peak_level(full_filepath_with_extension):
    """
    Calculates the peak audio level (max_dBFS) of an audio file.
    Expects the full path including the .mp3 extension.
    """
    try:
        # Load directly using the provided full path
        audio = AudioSegment.from_mp3(full_filepath_with_extension)
        if audio.duration_seconds > 0:
            peak_dbfs = audio.max_dBFS
            if peak_dbfs == -math.inf:
                 # Use the actual filename in the warning
                 # print(f"Warning: File {os.path.basename(full_filepath_with_extension)} appears to be silent.")
                 return -100.0
            return peak_dbfs
        else:
            # print(f"Warning: File {os.path.basename(full_filepath_with_extension)} has zero duration.")
            return -100.0
    # More specific error messages
    except CouldntDecodeError:
        print(f"Error: Could not decode file: {os.path.basename(full_filepath_with_extension)}. Skipping analysis.")
        return None
    except FileNotFoundError:
        # This shouldn't happen if pre-checked, but good practice
        print(f"Error: File not found during analysis: {full_filepath_with_extension}. Skipping.")
        return None
    except Exception as e:
        print(f"Error analyzing peak level for {os.path.basename(full_filepath_with_extension)}: {e}. Skipping analysis.")
        return None

def calculate_volume_scale(target_peak_dbfs, current_peak_dbfs):
    """Calculates the pygame volume scale factor (0.0 to 1.0)."""
    if target_peak_dbfs is None or current_peak_dbfs is None:
        return 0.5 # Default volume if analysis failed

    if target_peak_dbfs <= -100.0 or current_peak_dbfs <= -100.0:
         return 0.5

    db_difference = target_peak_dbfs - current_peak_dbfs
    scale_factor = 10**(db_difference / 5.0) # Use 20 for peak normalization
    scaled_volume = max(0.0, min(1.0, scale_factor))
    return scaled_volume

# --- Main Combined Logic ---

def main(target_mood_str, top_n, mp3_folder_path, tags_file_path, sample_file_name, logging=False):
    # === Part 1: Find Top N Files Matching Mood ===

    print("--- Mood Matching Phase ---")
    # --- Load Tags Data ---
    if not os.path.exists(tags_file_path):
         print(f"Error: Tags file not found at '{tags_file_path}'")
         sys.exit(1)
    try:
        with open(tags_file_path) as f:
            # Ensure tags_data is treated as a dictionary
            tags_data = json.load(f)
            if not isinstance(tags_data, dict) or not tags_data:
                print(f"Error: Tags data file '{tags_file_path}' is empty or not a valid JSON object.")
                sys.exit(1)
            print(f"Loaded {len(tags_data)} tag data items from '{tags_file_path}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{tags_file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tags file '{tags_file_path}': {e}")
        sys.exit(1)

    # --- Load Model ---
    print(f"Loading sentence transformer model '{MODEL_NAME}' onto device '{DEVICE}'...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        print("Please ensure sentence-transformers and its dependencies (like PyTorch/TensorFlow) are installed.")
        sys.exit(1)

    # --- Prepare Strings and Generate Embeddings ---
    strings_to_encode = []
    # Map embedding index to original filename *without* extension (as in tags.json)
    original_indices = {}
    filenames_in_tags = list(tags_data.keys()) # These are the keys from JSON (no extension)

    if logging:
        print("\nProcessing mood tags...")

    current_index = 0
    valid_filenames_for_embedding = [] # Keep track of filenames we actually embed
    for filename_base in filenames_in_tags: # filename_base has no extension
        # Construct the full path to check if the actual MP3 file exists
        potential_mp3_path = os.path.join(mp3_folder_path, filename_base + ".mp3")

        mood_list = tags_data.get(filename_base, [])

        # Only proceed if the MP3 file exists and has tags
        if os.path.isfile(potential_mp3_path):
            if mood_list:
                shuffled_moods = mood_list[:]
                random.shuffle(shuffled_moods)
                combined_mood_string = ", ".join(shuffled_moods)

                strings_to_encode.append(combined_mood_string)
                original_indices[current_index] = filename_base # Store base name
                valid_filenames_for_embedding.append(filename_base)
                current_index += 1
            elif logging:
                 print(f"  Skipping '{filename_base}': No mood tags found (file exists).")
        elif logging:
            print(f"  Skipping '{filename_base}': Corresponding MP3 file not found at '{potential_mp3_path}'.")


    if not strings_to_encode:
        print("Error: No valid MP3 files with tags found to process.")
        sys.exit(1)

    # Add the target string
    strings_to_encode.append(target_mood_str)
    target_label = f"Target: {target_mood_str}"
    target_index = len(strings_to_encode) - 1
    original_indices[target_index] = target_label # Special label for target

    if logging:
        print(f"\nEncoding {len(strings_to_encode)} strings (target + {len(valid_filenames_for_embedding)} files)...")

    embeddings = model.encode(["music that is " + x for x in strings_to_encode], show_progress_bar=logging)
    print("Embeddings generated.")

    # --- Calculate Similarities ---
    target_embedding = embeddings[target_index]
    other_embeddings = []
    # other_labels will store base filenames (no extension)
    other_labels = []
    for i in range(len(embeddings)):
        if i != target_index:
            other_embeddings.append(embeddings[i])
            # Get the original base filename corresponding to this embedding index
            other_labels.append(original_indices[i])

    if not other_labels:
        print("Error: No file embeddings generated to compare against target (problem after filtering?).")
        sys.exit(1)

    similarities = cosine_similarity([target_embedding], other_embeddings)[0]
    # Pair base filenames with scores
    results = list(zip(other_labels, similarities))
    results_sorted = sorted(results, key=lambda item: item[1], reverse=True)

    # --- Get Top N Results (Base Filenames and Scores) ---
    top_results = results_sorted[:top_n]

    if not top_results:
        print(f"No files found matching the mood '{target_mood_str}'.")
        sys.exit(1)

    # Print the mood matching results table
    print("\n--- Top Mood Matches ---")
    headers = ["#", "File (Base Name)", "Similarity"]
    table_data = []
    # Store the *full paths with extension* of the top files for playback
    top_files_full_paths_playback = []
    for rank, (base_filename, score) in enumerate(top_results):
        table_data.append([rank + 1, base_filename, f"{score:.4f}"])
        # Construct the full path *with extension* for the playback list
        full_path = os.path.join(mp3_folder_path, base_filename + ".mp3")
        top_files_full_paths_playback.append(full_path)

    print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

    # === Part 2: Volume Normalization and Playback ===

    print("\n--- Volume Normalization and Playback Phase ---")
    # --- Basic Setup and File Checks ---
    if not os.path.isdir(mp3_folder_path):
        print(f"Error: MP3 folder not found: '{mp3_folder_path}'")
        sys.exit(1)

    # Construct the full path for the sample file
    sample_filepath_full = os.path.join(mp3_folder_path, sample_file_name)
    if not os.path.isfile(sample_filepath_full):
        print(f"Error: Sample MP3 for volume reference not found: '{sample_filepath_full}'")
        sys.exit(1)

    # --- Initialize Pygame ---
    print("Initializing Pygame Mixer...")
    try:
        # Try common settings first
        try:
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.init()
            pygame.mixer.init()
        except pygame.error:
             print("Standard pygame init failed, trying frequency 22050...")
             pygame.mixer.pre_init(22050, -16, 2, 2048) # Try lower frequency
             pygame.init()
             pygame.mixer.init() # Try initializing again
        print("Pygame initialized successfully.")
    except pygame.error as e:
        print(f"Error initializing Pygame: {e}")
        print("Ensure audio drivers are installed and configured. Try different pre_init parameters if necessary.")
        sys.exit(1)

    # --- Analyze Required Audio Files ---
    # Files to analyze are the top N selected ones + the sample file
    # These paths already include the .mp3 extension
    files_to_analyze = list(set(top_files_full_paths_playback + [sample_filepath_full]))

    print(f"\nAnalyzing {len(files_to_analyze)} audio files for peak levels...")
    # Store analysis results using the *full path with extension* as the key
    audio_data = {}

    for full_path in files_to_analyze:
        # Double-check existence before analyzing (should exist based on earlier checks, but safer)
        if not os.path.isfile(full_path):
            print(f"Warning: File path expected for analysis does not exist: '{full_path}'. Skipping.")
            continue

        filename_ext = os.path.basename(full_path) # Filename with extension
        if logging:
            print(f"  Analyzing: {filename_ext}")

        # Pass the full path with extension to the analysis function
        peak_dbfs = get_audio_peak_level(full_path)

        if peak_dbfs is not None:
            # Use the full path with extension as the key
            audio_data[full_path] = {'peak_dbfs': peak_dbfs, 'scale': 1.0} # Default scale
        # Error message already printed inside get_audio_peak_level if analysis failed

    if not audio_data:
        print("Error: No audio files could be successfully analyzed.")
        pygame.quit()
        sys.exit(1)

    # --- Calculate Target Peak Level and Scaling Factors ---
    if sample_filepath_full not in audio_data:
         print(f"\nError: Sample file '{sample_file_name}' was found but could not be analyzed (check logs above).")
         pygame.quit()
         sys.exit(1)

    target_peak_dbfs = audio_data[sample_filepath_full]['peak_dbfs']
    print(f"\nReference Peak Level (from {sample_file_name}): {target_peak_dbfs:.2f} dBFS")

    if target_peak_dbfs is None or target_peak_dbfs <= -100.0:
        print("Error: Could not determine a valid peak level for the sample file. Cannot scale.")
        pygame.quit()
        sys.exit(1)

    print("\nCalculating volume scales for the top files:")
    # Calculate scales ONLY for the files we intend to play (the top N)
    # Iterate using the playback list which has full paths with extensions
    for full_path in top_files_full_paths_playback:
        filename_ext = os.path.basename(full_path)
        # Check if analysis data exists for this full path
        if full_path in audio_data:
            data = audio_data[full_path]
            scale = calculate_volume_scale(target_peak_dbfs, data['peak_dbfs'])
            # Update scale in our data dict using the full path key
            audio_data[full_path]['scale'] = scale
            if logging:
                 print(f"  - {filename_ext}:")
                 print(f"      Peak Level: {data['peak_dbfs']:.2f} dBFS -> Scale: {scale:.3f}")
        else:
             # This case means the file was in top_results but analysis failed/skipped
             print(f"  - Skipping scale calculation for {filename_ext} (analysis failed or file missing). Will use default scale 0.5.")
             # Add placeholder data if missing for this full_path key
             audio_data[full_path] = {'peak_dbfs': None, 'scale': 0.5}


    # --- Prepare Playback List (already have top_files_full_paths_playback) ---
    playback_list = top_files_full_paths_playback[:] # Copy the list of full paths

    # --- Playback Loop ---
    print(f"\n--- Starting Looping Playback of Top {len(playback_list)} Mood Matches (Ctrl+C to stop) ---")
    try:
        while True: # Loop indefinitely until interrupted
            random.shuffle(playback_list) # Shuffle the order for each pass

            for full_path in playback_list:
                filename_ext = os.path.basename(full_path)

                # Lookup analysis data using the full_path key
                if full_path not in audio_data:
                    print(f"\nSkipping {filename_ext}: Missing analysis data.")
                    continue # Should not happen if placeholder was added

                # Use calculated scale, default to 0.5 if necessary (e.g., from placeholder)
                volume_scale = audio_data[full_path].get('scale', 0.5) # Default added protection
                original_peak_dbfs = audio_data[full_path].get('peak_dbfs', 'N/A')
                peak_str = f"{original_peak_dbfs:.2f}" if isinstance(original_peak_dbfs, float) else original_peak_dbfs

                print(f"\n▶️ Playing: {filename_ext}")
                print(f"  Original Peak: {peak_str} dBFS | Target Scale: {volume_scale:.3f}")

                try:
                    # Load sound using the full path with extension
                    sound = pygame.mixer.Sound(full_path)
                    sound.set_volume(volume_scale)
                    sound.play()

                    # Wait for the sound to finish playing + handle events
                    while pygame.mixer.get_busy():
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                print("\nQuit event detected. Stopping playback.")
                                pygame.mixer.stop()
                                raise SystemExit
                        pygame.time.Clock().tick(10) # Prevent high CPU usage

                except pygame.error as e:
                    print(f"  Error playing {filename_ext}: {e}")
                    # Add a small delay before trying the next track on error
                    time.sleep(1)
                except Exception as e:
                     print(f"  An unexpected error occurred during playback of {filename_ext}: {e}")
                     time.sleep(1)

    except SystemExit:
        print("Playback stopped by user.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user (Ctrl+C).")
    finally:
        print("\nPlayback finished or stopped. Cleaning up...")
        pygame.mixer.quit()
        pygame.quit()
        print("Done.")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and play music files matching a specified mood with volume normalization.')
    parser.add_argument('-mood', '--mood', type=str, required=True,
                        help='Target mood description (e.g., "epic space battle", "calm ambient study").')
    parser.add_argument('-top', '--top', type=int, default=5,
                        help='Number of top matching files to find and play.')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging output.')
    # Use arguments for paths to make it more flexible
    parser.add_argument('--folder', type=str, default=MP3_FOLDER,
                        help=f'Path to the MP3 folder (default: {MP3_FOLDER}).')
    parser.add_argument('--tags', type=str, default=TAGS_FILE,
                        help=f'Path to the tags JSON file (default: {TAGS_FILE}).')
    parser.add_argument('--sample', type=str, default=SAMPLE_MP3_FILENAME,
                        help=f'Filename (with extension) of the reference volume MP3 inside the MP3 folder (default: {SAMPLE_MP3_FILENAME}).')

    args = parser.parse_args()

    # --- Dependency Check ---
    try:
        import sentence_transformers
        import sklearn
        import pydub
        import pygame
        from tabulate import tabulate
        import os
        # torch already imported for device check
    except ImportError as e:
        print(f"Error: Missing required Python package: {e.name}")
        print("Please install required packages: pip install sentence-transformers scikit-learn pydub pygame tabulate torch")
        print("Note: pydub requires FFmpeg/libav. Pygame might have OS dependencies.")
        sys.exit(1)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Call the main function with potentially overridden paths
    main(args.mood, args.top,
         mp3_folder_path=args.folder,
         tags_file_path=args.tags,
         sample_file_name=args.sample,
         logging=args.log)
