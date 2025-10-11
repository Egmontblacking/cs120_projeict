
    try:
        playback_data = playback_queue.get_nowait()
        if playback_data is None:
            print("Playback finished.")
        else:
            out1_array[:] = playback_data
    except queue.Empty:
        out1_array[:] = 0
