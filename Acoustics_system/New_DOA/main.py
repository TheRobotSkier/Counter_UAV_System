import numpy as np
import soundfile as sf
import time

from simulation import (
    regular_tetrahedron_array,
    cart2sph,
    compute_travel_times,
    plot_array_and_source,
    load_soundfile,
    make_all_channels_same,
    fractional_delay_fd,
    delay_channel_soundfile
)

from srp_doa import (
    precompute_tau_grid,
    srp_phat_precompute,
    compute_srp_map,
    find_best_direction,
    plot_srp_map
)

from bandpass_FIR import FIRBandpass4Ch
from bandpass_filtfilt import design_bandpass, apply_bandpass

# ---------------------------
# Parameters
# ---------------------------

D_M = 0.5#1.0  # distance between microphones (meters)
SPEED_OF_SOUND = 343.0  # m/s

# Frame length and overlap:
FRAME_DUR_SEC = 0.100  # seconds
OVERLAP_50 = True  # 50% overlap if True, else no overlap

# Bandpass filter settings
BANPASS_FILTER=False #False # keep false it do funny stuff
LOWCUT = 100.0   # Hz
HIGHCUT = 8000.0 # Hz

# SRP grid resolution
# Search grid
AZ_DEG_STEP = 1.0     # azimuth step [deg]
EL_DEG_STEP = 1.0     # elevation step [deg]
AZIMUTHS = np.arange(0.0, 360.0, AZ_DEG_STEP)       # 0..360 exclusive
ELEVATIONS = np.arange(0.0, 90.0 + 1e-6, EL_DEG_STEP)  # 0..90 inclusive

# GCC-PHAT interpolation factor (higher -> finer delay resolution, more CPU)
INTERP_GCC = 16

# Simulation settings
SIMULATION_MODE = True         # Toggle real/simulated audio
SOURCE_POS = np.array([20.0, 100.0, 20.5])  # [m] source for simulation


# File mode settings
WAV_FILENAME = "Acoustics_system/Recordings/Four_mic_recordings/Re-recording_of_Phantom_Test_File1.wav"
SKIP_SECONDS = 1.0
ALL_CHANNELS_SAME= False  # make all channels same (for testing)
CHANNEL_INDEX = 0      # which channel to use if ALL_CHANNELS_SAME=True [0..3]




def main():
    # Build mic array
    mic_positions = regular_tetrahedron_array(D_M)  # [4,3]



    if SIMULATION_MODE:
        # Compute travel times and TDOAs
        distances, times, relative_times, pairs, pair_tdoas = compute_travel_times(
            SOURCE_POS, mic_positions, SPEED_OF_SOUND
        )
        print("Simulated source position (m):", SOURCE_POS)
        dist_gt,az_gt,el_gt=np.round(cart2sph(SOURCE_POS), 2)
        print(f"Sound source ground truth (spherical):azimuth={az_gt} deg, elevation={el_gt} deg, distance={dist_gt} m")

        print("\nRelative arrival times [ms] (0 = first detection):")
        for (i, j), dt in pair_tdoas.items():
            print(f"  MIC_{i+1}-MIC_{j+1}: {dt * 1e3:.3f} ms")

        # Load simulated audio file
        data, fs = load_soundfile(WAV_FILENAME)

        # Optionally make all channels the same (for testing)
        if ALL_CHANNELS_SAME:
            data = make_all_channels_same(data, CHANNEL_INDEX)
            print(f"All channels set to channel {CHANNEL_INDEX} for testing.")

        # Apply delays to simulate TDOAs
        delayed_data = delay_channel_soundfile(data, fs, relative_times)

        # Initialize bandpass filter
        if BANPASS_FILTER:
            sos = design_bandpass(LOWCUT, HIGHCUT, fs)
            #bandpass_filter = FIRBandpass4Ch(lowcut=LOWCUT,highcut=HIGHCUT,sr=fs)

        # Visualize array and source
        plot_array_and_source(mic_positions, SOURCE_POS)

        # Precompute TDOA grid
        tau_grid, max_tdoa_sec = precompute_tau_grid(
            mic_positions,
            pairs,
            AZIMUTHS,
            ELEVATIONS,
            SPEED_OF_SOUND
        )

        # Main processing loop (simulate 10 Hz updates)

        FRAME_LEN = int(fs * FRAME_DUR_SEC)
        if OVERLAP_50:
            HOP_LEN = FRAME_LEN // 2      # 50% overlap
        else:
            HOP_LEN = FRAME_LEN          # no overlap

        num_frames = (delayed_data.shape[0] - FRAME_LEN) // HOP_LEN

        print(f"Running SRP-PHAT on {num_frames} frames, hop={HOP_LEN} samples:")

        # Storage for timing statistics
        frame_times = []

        for frame_idx in range(num_frames):
            start = frame_idx * HOP_LEN
            end   = start + FRAME_LEN

            frame = delayed_data[start:end, :]

            # Apply bandpass filter
            if BANPASS_FILTER:
                frame_filtered = apply_bandpass(frame, sos)
                #frame_filtered = bandpass_filter.process(frame)
            else:
                frame_filtered = frame

            # start timer
            t0 = time.perf_counter()

            # GCC-PHAT for all pairs (per frame)
            pair_cc, pair_lags = srp_phat_precompute(
                frame_filtered,
                fs,
                pairs,
                max_tdoa_sec,
                interp=INTERP_GCC
            )
            # Compute SRP map
            srp_map = compute_srp_map(
                pair_cc,
                pair_lags,
                tau_grid
            )
            # Find best direction
            best_az, best_el = find_best_direction(srp_map, AZIMUTHS, ELEVATIONS)

            # stop timer
            t1 = time.perf_counter()
            dt = (t1 - t0) * 1000   # ms
            frame_times.append(dt)

            print(f"Frame {frame_idx:03d}: DOA → az={best_az:.1f}°, el={best_el:.1f}° | dt: {dt:.2f} ms")

            if frame_idx == 0:
                # Plot SRP map for first frame
                plot_srp_map(srp_map, AZIMUTHS, ELEVATIONS, best_az, best_el)
        
        # print midian time
        median_time = np.median(frame_times)
        print(f"\nMedian processing time per frame: {median_time:.2f} ms") 
        mean_time = np.mean(frame_times)
        print(f"Mean processing time per frame: {mean_time:.2f} ms")
        #print max processing time
        print(f"Max processing time per frame: {np.max(frame_times):.2f} ms")
        # print frequency
        print(f"Estimated processing frequency: {1000 / mean_time:.2f} Hz")


    else:
        print("Real time audio mode not yet implemented.")
        return
    
    
if __name__ == "__main__":
    main()