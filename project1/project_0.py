import jack
import threading
import numpy as np
import queue
import soundfile as sf
import scipy
from Transmitter import Transmitter
from Receiver import Receiver
from CONFIG import Config
import time

event = threading.Event()
stop_event = threading.Event()

in_queue = queue.Queue()
playback_queue = queue.Queue()

client = jack.Client("PythonAudioProcessor")
print("JACK server sample rate:", client.samplerate)
in1 = client.inports.register("input_1")
out1 = client.outports.register("output_1")

CHANNELS = 1

# print(client.blocksize)

# $$ f(t)=\sin (2 \pi \cdot 1000 \cdot t)+\sin (2 \pi \cdot 10000 \cdot t) $$
task2_wave_component = [
    {"A": 1.0, "f": 1000.0, "phi": 1.0},
    {"A": 1.0, "f": 10000.0, "phi": 1.0},
]

C_E_G = [
    {"A": 1.0, "f": 261.63, "phi": 1.0},  # C4
    {"A": 1.0, "f": 329.63, "phi": 1.0},  # E4
    {"A": 1.0, "f": 392.00, "phi": 1.0},  # G4
]

default_wave_component = [{"A": 1.0, "f": 440.0, "phi": 1.0}]
global_phase_accumulator = 0


######################
def transmitter_thread_func():
    """Reads file, creates frame, and puts it on the playback queue."""
    print("--- Transmitter Thread Started ---")
    try:
        # Generate random bits for INPUT.txt
        bits = np.random.randint(0, 2, Config.PAYLOAD_BITS, dtype=np.uint8)
        with open(Config.INPUT_FILENAME, "w") as f:
            f.write("".join(map(str, bits)))
        print(
            f"Generated {Config.PAYLOAD_BITS} random bits in '{Config.INPUT_FILENAME}'"
        )

        tx = Transmitter(Config())
        frame_signal = tx.create_frame(bits)

        # Put frame signal onto playback queue block by block
        frame_size = client.blocksize
        current_pos = 0
        while not stop_event.is_set() and current_pos < len(frame_signal):
            end_pos = current_pos + frame_size
            block = frame_signal[current_pos:end_pos]

            if len(block) < frame_size:  # Pad the last block with zeros
                block = np.pad(block, (0, frame_size - len(block)), "constant")

            playback_queue.put(block)
            current_pos += frame_size

        print("--- Transmission frame queued successfully ---")

    except Exception as e:
        print(f"Transmitter thread error: {e}")
    finally:
        print("--- Transmitter Thread Finished ---")


########################
def receiver_thread_func():
    """Gets audio from recording queue and processes it to find frames."""
    print("--- Receiver Thread Started ---")
    rx = Receiver(Config())
    total_bits_received = 0
    min_processing_unit = np.array([], dtype=np.float32, len=0)
    min_processing_len = client.blocksize * 10  # Process at least 10 blocks at a time

    try:
        with open(Config.OUTPUT_FILENAME, "w") as f:
            while not stop_event.is_set():
                # try:
                while not in_queue.empty():
                    min_processing_unit = np.append(
                        min_processing_unit, in_queue.get_nowait()
                    )

    except Exception as e:
        print(f"Receiver thread error: {e}")
    finally:
        print("--- Receiver Thread Finished ---")


def sinwave_player_thread_func(wave: list[dict] = default_wave_component, volume=10.0):

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
        # print("block size:", frame_size, "len data:", total_frames)
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
        print("Audio file has been put into queue.")


def record_thread_func(OUTPUT_FILENAME="recording.wav"):
    try:
        with sf.SoundFile(
            OUTPUT_FILENAME, mode="w", samplerate=client.samplerate, channels=CHANNELS
        ) as file:
            while not stop_event.is_set():
                try:
                    data = in_queue.get(timeout=0.5)
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

    # if "record_thread" in globals() and record_thread.is_alive():
    in_queue.put(in1_array.copy())

    try:
        playback_data = playback_queue.get_nowait()
        out1_array[:] = playback_data
    except queue.Empty:
        out1_array[:] = 0


with client:
    print("Python JACK client is running.")
    client.connect("system:capture_1", in1)
    client.connect(out1, "system:playback_1")

    try:
        while True:
            user_input = (
                input(
                    "Enter 'r' to record, "
                    "'p' to play wave from audio file, "
                    "'task2' to play custom wave, "
                    "'tx' to transmit, "
                    "'rx' to receive:"
                )
                .strip()
                .lower()
            )
            match user_input:
                case "r":
                    record_thread = threading.Thread(target=record_thread_func)
                    record_thread.start()
                case "p":
                    reader_thread = threading.Thread(target=audio_player_thread_func)
                    reader_thread.start()
                case "tx":
                    tx_thread = threading.Thread(target=transmitter_thread_func)
                    tx_thread.start()
                case "task2":
                    wave_thread = threading.Thread(
                        target=sinwave_player_thread_func,
                        args=(C_E_G, 5.0),
                    )
                    wave_thread.start()
            # event.wait()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop_event.set()
