# DT2119, Lab 1 Feature Extraction
from lab1_tools import *
from scipy import fftpack
import math

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps) 
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    result = []
    for i in range(0, len(samples)-1 - winshift, winshift):
        window = []
        for j in range(0, winlen):
            if i+j < len(samples):
                window.append(samples[i+j])

        if(i + j) < len(samples):      
          result.append(window)

    return np.asarray(result)
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    a = [1]
    b = [1, -p]
    return signal.lfilter(b, a, input)
    
def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    hamming = signal.hamming(input.shape[1], sym=0)

    #print("hamming", hamming.shape)
    #print("input", input.shape)
    out = input * hamming
    #print("out", out.shape)
    return  input * hamming  #np.dot(input, hamming)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fft = fftpack.fft(input, n=nfft, axis=-1)
    #print(fft.shape)
    result = np.abs(fft) ** 2
    #print(result.shape)
    return result


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = input.shape[1]
    fbank = trfbank(samplingrate, nfft)

    return np.log(input.dot(fbank.T))


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    cepstral = fftpack.dct(input)

    return cepstral[:,:13]


def euclideanDistance(digit1, digit2):

    N = digit1.shape[0]
    M = digit2.shape[0]
    distance = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            distance[n,m] = np.linalg.norm(digit1[n] - digit2[m])
    return distance

def helper_dtw(utterance1,utterance2):
    N = len(utterance1)
    M = len(utterance2)

    #create an empty distance matrix with high values
    Acc_distance = np.zeros((N,M))

    local_distance = euclideanDistance(utterance1, utterance2)
    Acc_distance[0,0] = 0

    for i in range(1, N):
        for j in range(1, M):
            #cost = dist(x[i], y[j])
            #local_distance = euclideanDistance(utterance1, utterance2)
            Acc_distance[i,j] = local_distance[i,j] + min(Acc_distance[i-1,j], Acc_distance[i,j-1], Acc_distance[i-1,j-1])

    return Acc_distance[N-1, M-1]

def dtw(x, y):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    print("in dtw, vector lengths: ")
    global_distance = helper_dtw(x, y)

    return global_distance / (len(x) + len(y))
