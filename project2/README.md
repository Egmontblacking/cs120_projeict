# Project2

- [x] Task 1: (4 points) Cable Connection

Usage: use tx to encode and generate audio, rx to receive and decode the audio.

Implementation: As cabled transmission is way more stable than transmitting in air, we use OOK instead of PSK. Actually, according to our experiments, PSK is not stable even when cabled as a result of carrier phase misalignment, so it should be avoid using unless paired with using orthogonal carrier waves or actively recovering the accurate start of the received signal. In our implementation of OOK, we use 500bits/packet and 8 bit of crc, with a preamble of 440bits.

TODO: 

- [ ] Use 4b5b encoding to know the exact end of the signal and further improve efficiency.

- [ ] Task 2: (5 points) Acknowledgement

- [ ] Task 3: (2 points) Carrier Sense Multiple Access

- [ ] Task 4: (1 point) CSMA with Interference