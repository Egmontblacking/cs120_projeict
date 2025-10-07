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

OUTPUT_FILENAME = "recording.wav"
INPUT_FILENAME = "predefined_wave.wav"
CHANNELS = 1

print(client.blocksize)


def audio_reader_thread_func():
    try:
        data, samplerate = sf.read(INPUT_FILENAME, dtype="float32")
        frame_size = client.blocksize
        total_frames = len(data)
        print("block size:", frame_size, "len data:", total_frames)
        current_frame = 0

        if samplerate != client.samplerate:
            # TODO: unable to change JACK server sample rate, ref: https://www.reddit.com/r/linuxaudio/comments/195231g/qjackctl_does_not_change_the_samplerate_of_jack/
            raise ValueError(
                f"文件采样率 ({samplerate} Hz) 与 JACK 服务器采样率 "
                f"({client.samplerate} Hz) 不匹配"
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
        print(f"Audio reader thread error: {e}")
    finally:
        print("Audio reader thread finished.")


def record_thread_func():
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
                input("Enter 'r' to record, 'p' to play wave: ").strip().lower()
            )
            match user_input:
                case "r":
                    record_thread = threading.Thread(target=record_thread_func)
                    record_thread.start()
                    break
                case "p":
                    reader_thread = threading.Thread(target=audio_reader_thread_func)
                    reader_thread.start()
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
