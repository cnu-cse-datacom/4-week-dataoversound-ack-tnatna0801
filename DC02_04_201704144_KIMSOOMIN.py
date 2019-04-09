from __future__ import print_function

import sys
import wave

from io import StringIO

import alsaaudio
import colorama
import numpy as np
import pyaudio

from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format
from time import sleep

HANDSHAKE_START_HZ = 4096
HANDSHAKE_END_HZ = 5120 + 1024

START_HZ = 1024
STEP_HZ = 256
BITS = 4

FEC_BYTES = 4

MY_STUDENTID = "201704144" 


def stereo_to_mono(input_file, output_file):
    inp = wave.open(input_file, 'r')
    params = list(inp.getparams())
    params[0] = 1 # nchannels
    params[3] = 0 # nframes

    out = wave.open(output_file, 'w')
    out.setparams(tuple(params))

    frame_rate = inp.getframerate()
    frames = inp.readframes(inp.getnframes())
    data = np.fromstring(frames, dtype=np.int16)
    left = data[0::2]
    out.writeframes(left.tostring())

    inp.close()
    out.close()

def yield_chunks(input_file, interval):
    wav = wave.open(input_file)
    frame_rate = wav.getframerate()

    chunk_size = int(round(frame_rate * interval))
    total_size = wav.getnframes()

    while True:
        chunk = wav.readframes(chunk_size)
        if len(chunk) == 0:
            return

        yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
    w = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk))
    peak_coeff = np.argmax(np.abs(w))
    peak_freq = freqs[peak_coeff]
    return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
    return abs(freq1 - freq2) < 20

def decode_bitchunks(chunk_bits, chunks):
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        to_fill = min(bits_left, can_fill)
        offset = chunk_bits - next_read_bit - to_fill
        byte <<= to_fill
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        byte |= shifted >> offset;
        bits_left -= to_fill
        next_read_bit += to_fill
        if bits_left <= 0:

            out_bytes.append(byte)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits
        #print(out_bytes)
    return out_bytes

def decode_file(input_file, speed):
    wav = wave.open(input_file)
    if wav.getnchannels() == 2:
        mono = StringIO()
        stereo_to_mono(input_file, mono)

        mono.seek(0)
        input_file = mono
    wav.close()

    offset = 0
    for frame_rate, chunk in yield_chunks(input_file, speed / 2):
        dom = dominant(frame_rate, chunk)
        print("{} => {}".format(offset, dom))
        offset += 1

def extract_packet(freqs):
    freqs = freqs[::2]
    #print (freqs)
    bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
    #print (bit_chunks)
    bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
    #print (bit_chunks)
    return bytearray(decode_bitchunks(BITS, bit_chunks))

#make frequency
def compress_packet(bit_chunk):
    freqs = []
    for i in range(len(bit_chunk)):
        freqs.append((bit_chunk[i] * STEP_HZ) + START_HZ)
    freqs.insert(0, HANDSHAKE_START_HZ) # put HANDSHAKE_START_HZ at the first
    freqs.append(HANDSHAKE_END_HZ) # put HANDSHAKE_END_HZ at the end of list
    #print(freqs)
    return freqs

 
#1byte ascii to 4bits 4bits
def byte_to_bits(byte_stream):
    bit_stream = []
    for i in range(len(byte_stream)):
        bit_stream.append(byte_stream[i]//(2**BITS))
        bit_stream.append(byte_stream[i]%(2**BITS))

        # need to twice???
    #print(len(bit_stream))
    #print(bit_stream)
    return bit_stream
     

def display(s):
    cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')


def speak_linux(byte_stream, frame_rate=44100):
    p = pyaudio.PyAudio() 
    bit_chunk = byte_to_bits(byte_stream)
    #print(bit_chunk)
    freq = compress_packet(bit_chunk)
    #print(freq) #test

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output = True)

    #volume = 0.5
    duration = 0.5
    for i in range(len(freq)): 
        samples = (np.sin(2*np.pi*np.arange(frame_rate*duration)*(freq[i]/frame_rate))).astype(np.float32)
        stream.write(samples)
        #print("arange: " + str(np.arange(frame_rate*duration)*(freq[i]/frame_rate)))
        print("freq : " + str(freq[i]))
        
    #print(samples)
    #stream.write(volume*samples)
    
    stream.stop_stream()
    stream.close()

    p.terminate()

def listen_linux(frame_rate=44100, interval=0.1):

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="default")
    mic.setchannels(1)
    mic.setrate(44100)
    mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    num_frames = int(round((interval / 2) * frame_rate))
    mic.setperiodsize(num_frames)
    print("start...")

    in_packet = False
    packet = []

    while True:
        l, data = mic.read()
        if not l:
            continue

        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant(frame_rate, chunk)

        if in_packet and match(dom, HANDSHAKE_END_HZ):
            byte_stream = extract_packet(packet)
            print("end") 
            try:
                byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                #print(byte_stream) #test
                byte_stream = byte_stream.decode("utf-8")
                print(byte_stream) #test
                if MY_STUDENTID in byte_stream:
 
                    re_byte_stream = byte_stream[10:] #without STUDENTID
                    display(re_byte_stream)
                    re_byte_stream = re_byte_stream.encode("utf-8")
                    re_byte_stream = RSCodec(FEC_BYTES).encode(re_byte_stream)
                    
                    ascii_temp = byte_stream.encode("utf-8") # with STUDENTID
                    ascii_temp = RSCodec(FEC_BYTES).encode(ascii_temp)
                    print_ascii = []
                    for num in range(len(ascii_temp)): 
                        print_ascii.append(ascii_temp[num])
                    print(print_ascii)  
                    
                    sleep(1)
                    
                    #print sound
                    speak_linux(re_byte_stream)
            
            except ReedSolomonError as e:
                pass
                #print("{}: {}".format(e, byte_stream))

            packet = []
            in_packet = False
        elif in_packet:
            packet.append(dom)
        elif match(dom, HANDSHAKE_START_HZ):
            in_packet = True
            print("strat HandShake")

if __name__ == '__main__':
	colorama.init(strip=not sys.stdout.isatty())

	#decode_file(sys.argv[1], float(sys.argv[2]))
	listen_linux()
