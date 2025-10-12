import jack
import threading
import numpy as np
import queue
import soundfile as sf
import scipy
from Transmitter import Transmitter
from Receiver import Receiver, ReceiverState
from CONFIG import Config
import time


event = threading.Event()
stop_event = threading.Event()

record_queue = queue.Queue()
playback_queue = queue.Queue()

client = jack.Client("PythonAudioProcessor")
in1 = client.inports.register("input_1")
out1 = client.outports.register("output_1")

CONFIG = Config(client.samplerate, client.blocksize)


def playback_thread_func(data):
    try:
        frame_size = client.blocksize
        total_frames = len(data)
        current_frame = 0

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
        print("Audio file has been put into queue.\n")


def record_thread_func(record=True):
    try:
        with sf.SoundFile(
            CONFIG.OUTPUT_WAV_FILENAME,
            mode="w",
            samplerate=CONFIG.SAMPLE_RATE,
            channels=CONFIG.CHANNELS,
        ) as file:
            while not stop_event.is_set():
                try:
                    record_block = record_queue.get(timeout=0.1)

                    # Convert to numpy array
                    arr = np.asarray(record_block, dtype=np.float32)

                    if record:
                        file.write(arr)
                    else:
                        # Pass to receiver
                        rx.receive_sample(arr)

                        # Save received data continuously (will be updated as frames arrive)
                        # This allows stopping at any time

                except queue.Empty:
                    continue
    except Exception as e:
        print(f"Recording thread error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Save received data when stopping (for rx mode)
        if not record and "rx" in globals():
            try:
                stats = rx.get_statistics()
                with open(CONFIG.OUTPUT_FILENAME, "w") as f:
                    f.write("".join(map(str, stats["all_bits"])))
                print(f"\n=== Reception Complete ===")
                print(f"Correct frames: {rx.correct_frames}/{rx.total_frames}")
                print(f"Total bits received: {len(stats['all_bits'])}")
                print(f"Saved to {CONFIG.OUTPUT_FILENAME}")
            except Exception as e:
                print(f"Error saving received data: {e}")
        print("Recording thread finished.")


@client.set_process_callback
def process(nframes):
    in1_array = in1.get_array()
    out1_array = out1.get_array()

    # if "record_thread" in globals() and record_thread.is_alive():
    record_queue.put(in1_array.copy())

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
                input("\nEnter 'r' to record, 'tx' to transmit, 'rx' to receive: ")
                .strip()
                .lower()
            )
            match user_input:
                case "r":
                    record_thread = threading.Thread(target=record_thread_func)
                    record_thread.start()
                    input("Recording... Press Enter to stop.\n")
                    stop_event.set()
                    record_thread.join()
                    stop_event.clear()

                case "tx":
                    # Read data from INPUT_FILENAME (must exist)
                    with open(CONFIG.INPUT_FILENAME, "r") as f:
                        bits = np.array(
                            list(map(int, f.read().strip())), dtype=np.uint8
                        )
                    print(f"Read {len(bits)} bits from {CONFIG.INPUT_FILENAME}")

                    # Verify length is multiple of FRAME_DATA_BITS
                    if len(bits) % CONFIG.FRAME_DATA_BITS != 0:
                        print(
                            f"Warning: Data length {len(bits)} is not a multiple of FRAME_DATA_BITS ({CONFIG.FRAME_DATA_BITS})"
                        )
                        print(
                            f"Expected multiple of {CONFIG.FRAME_DATA_BITS}, will use {len(bits) // CONFIG.FRAME_DATA_BITS} complete frames"
                        )

                    tx = Transmitter(CONFIG)
                    frame_signal = tx.create_frame(bits)

                    tx_thread = threading.Thread(
                        target=playback_thread_func, args=(frame_signal,)
                    )
                    tx_thread.start()
                    # tx_thread.join()
                    print("Transmission complete!")

                case "rx":
                    rx = Receiver(CONFIG)
                    rx_thread = threading.Thread(
                        target=record_thread_func, args=(False,)
                    )
                    rx_thread.start()
                    input("Receiving... Press Enter to stop.\n")
                    stop_event.set()
                    rx_thread.join()
                    stop_event.clear()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop_event.set()
