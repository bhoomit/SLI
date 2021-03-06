import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
import time

root = '.'

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = int(np.ceil( (len(samples) - frameSize) / float(hopSize))) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) #** factor

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
    scale *= (freqbins-1)/max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    # samples = samples[:, channel]
    s = stft(samples, binsize)

    # sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    # sshow = sshow[2:, :]
    sshow = s
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:256, :] # 0-11khz, ~10s interval
    #print "ims.shape", ims.shape

    image = Image.fromarray(ims)
    image = image.convert('L')
    image.save(name)


def process_single_file(source, target=None):
    if not target:
        target = '/tmp/{0}.png'.format(int(time.time() * 1000000))
    if '.wav' in source:
        wavfile = source
    else:
        wavfile = '/tmp/tmp.wav'
        os.system('mpg123 -w {0} {1}'.format(wavfile, source))
    plotstft(wavfile, name=target)
    if '.wav' not in source:
        os.remove(wavfile)
    return target


# def generate(files):
#     for iter, line in files: # first line of traininData.csv is header (only for trainingData.csv)
#         filepath, label = line.replace('\n', '').split(',')
#         filename = '{0}/{1}'.format(label, filepath[:-4])
#         target = os.path.join(root, 'data/train/' + filename + '.png')
#         source = os.path.join(root, 'data/voice/' + filepath)
#         if not os.path.exists(target):
#             process_single_file(source, target)
#         print("processed %d files" % (iter + 1))

def generate(root):
    for language in range(7):
        directory = os.path.join(root, 'voice/{}/'.format(language))
        count = int(len(os.listdir(directory)) * 0.8)
        counter = 0
        for filename in os.listdir(directory):
            source = os.path.join(directory, filename)
            if not os.path.isfile(source):
                continue
            folder = 'train' if count > 0 else 'validate'
            target = os.path.join(root, '{0}/{1}/'.format(folder, language), filename.replace('wav', 'png'))
            if not os.path.exists(target):
                process_single_file(source, target)
            count -= 1
            counter += 1
        print("processed %d files" % (counter))
"""
Way to invoke this

file = open(os.path.join(root, 'data/trainingData.csv'), 'r')
generate(enumerate(file.readlines()[1:]))

"""
