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
                    record_block = record_queue.get_nowait()
                    if record:
                        file.write(record_block)
                    else:
                        rx.receive_sample(record_block)
                        if rx.state == ReceiverState.RECEIVE_HEADER:
                            print("preamble detected")
                            break
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
                input("Enter 'r' to record, " "'tx' to transmit, " "'rx' to receive:")
                .strip()
                .lower()
            )
            match user_input:
                case "r":
                    record_thread = threading.Thread(target=record_thread_func)
                    record_thread.start()
                case "tx":
                    tx = Transmitter(CONFIG)
                    with open(Config.INPUT_FILENAME, "r") as f:
                        bits = np.array(
                            list(map(int, f.read().strip())), dtype=np.uint8
                        )
                    frame_signal = tx.create_frame(bits)
                    tx_thread = threading.Thread(
                        target=playback_thread_func, args=(frame_signal,)
                    )
                    tx_thread.start()
                case "rx":
                    rx = Receiver(CONFIG)
                    rx_thread = threading.Thread(
                        target=record_thread_func, args=(False,)
                    )
                    rx_thread.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop_event.set()
