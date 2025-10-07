import soundfile as sf
import numpy as np

duration = 1.0  # seconds
frequency = 440.0  # Hz (A4 note)
samplerate = 44100  # Hz
t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
silence = np.zeros(int(samplerate * duration))

block = np.concatenate((sine_wave, silence))
output = np.tile(block, 5)
sf.write("predefined_wave.wav", output, samplerate)
