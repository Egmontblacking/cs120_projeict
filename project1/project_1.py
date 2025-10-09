import jack
import threading
import numpy as np
import queue
import soundfile as sf

event = threading.Event()
stop_event = threading.Event()

recording_queue = queue.Queue()
playback_queue = queue.Queue(maxsize=1024)

client = jack.Client("PythonAudioProcessor")
in1 = client.inports.register("input_1")
out1 = client.outports.register("output_1")

CHANNELS = 1

# print(client.blocksize)

# $$ f(t)=\sin (2 \pi \cdot 1000 \cdot t)+\sin (2 \pi \cdot 10000 \cdot t) $$
task2_wave_component = [
    {"A": 1.0, "f": 1000.0, "phi": 1.0},
    {"A": 1.0, "f": 10000.0, "phi": 1.0},
]
default_wave_component = [{"A": 1.0, "f": 440.0, "phi": 1.0}]
global_phase_accumulator = 0


def wave_player_thread_func(wave: list[dict] = default_wave_component, volume=10.0):

    samplerate = client.samplerate
    blocksize = client.blocksize

    max_amp = sum(c["A"] for c in wave)
    normalization_factor = 1.0 / max_amp if max_amp > 0 else 0.0

    global global_phase_accumulator

    component_data = []
    for component in wave:
        # k = omega / samplerate
        k = 2.0 * np.pi * component["f"] / samplerate
        component_data.append({"k": k, "A": component["A"], "phi": component["phi"]})

    indices = np.arange(blocksize)

    try:
        while not stop_event.is_set():
            # phase_indices[n] = global_phase_accumulator + n
            phase_indices = global_phase_accumulator + indices

            total_wave = np.zeros(blocksize, dtype=np.float32)

            for data in component_data:
                wave = (
                    data["A"] * volume * np.sin(data["k"] * phase_indices + data["phi"])
                )
                total_wave += wave

            block_data = total_wave * normalization_factor

            playback_queue.put(block_data)

            global_phase_accumulator += blocksize

            # if playback_queue.full():
            #      time.sleep(0.001)

    except Exception as e:
        print(f"wave player thread error : {e}")
    finally:
        print("wave player thread finished.")


def audio_player_thread_func(INPUT_FILENAME="predefined_wave.wav"):
    try:
        data, samplerate = sf.read(INPUT_FILENAME, dtype="float32")
        frame_size = client.blocksize
        total_frames = len(data)
        print("block size:", frame_size, "len data:", total_frames)
        current_frame = 0

        if samplerate != client.samplerate:
            # can be fixed by `jackd -d coreaudio -r SAMPLERATE`
            raise ValueError(
                f"File sample rate ({samplerate} Hz) does not match JACK server sample rate "
                f"({client.samplerate} Hz)"
            )

        while not stop_event.is_set() and current_frame < total_frames:
            end_frame = min(current_frame + frame_size, total_frames)
            frame_data = data[current_frame:end_frame]
            if len(frame_data) < frame_size:
                frame_data = np.pad(
                    frame_data, (0, frame_size - len(frame_data)), "constant"
                )
            playback_queue.put(frame_data)
            current_frame += frame_size
    except Exception as e:
        print(f"Audio file player thread error: {e}")
    finally:
        print("Audio file player thread finished.")


def record_thread_func(OUTPUT_FILENAME="recording.wav"):
    try:
        with sf.SoundFile(
            OUTPUT_FILENAME, mode="w", samplerate=client.samplerate, channels=CHANNELS
        ) as file:
            while not stop_event.is_set():
                try:
                    data = recording_queue.get(timeout=0.5)
                    file.write(data)
                except queue.Empty:
                    continue
    except Exception as e:
        print(f"Recording thread error: {e}")
    finally:
        print("Recording thread finished.")


@client.set_process_callback
def process(nframes):

    in1_array = in1.get_array()
    out1_array = out1.get_array()

    # processed_data = in1_array * 0.5

    if "record_thread" in globals() and record_thread.is_alive():
        recording_queue.put(in1_array.copy())

    try:
        playback_data = playback_queue.get_nowait()
        if playback_data is None:
            print("Playback finished.")
        else:
            out1_array[:] = playback_data
    except queue.Empty:
        out1_array[:] = 0


with client:
    print(
        "Python JACK client is running. Press 'r' to start recording, Ctrl+C to stop."
    )
    client.connect("system:capture_1", in1)
    client.connect(out1, "system:playback_1")

    try:
        while True:
            user_input = (
                input(
                    "Enter 'r' to record, 'p' to play wave from audio file, 'w' to play custom wave: "
                )
                .strip()
                .lower()
            )
            match user_input:
                case "r":
                    record_thread = threading.Thread(target=record_thread_func)
                    record_thread.start()
                    break
                case "p":
                    reader_thread = threading.Thread(target=audio_player_thread_func)
                    reader_thread.start()
                case "task2":
                    wave_thread = threading.Thread(
                        target=wave_player_thread_func, args=(task2_wave_component)
                    )
                    wave_thread.start()
        event.wait()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop_event.set()
        if "record_thread" in locals() and record_thread.is_alive():
            record_thread.join()
            print("Recording completed.")
        if "reader_thread" in locals() and reader_thread.is_alive():
            reader_thread.join()
            print("Playback finished.")
        if "wave_thread" in locals() and wave_thread.is_alive():
            wave_thread.join()
            print("Wave playback finished.")
