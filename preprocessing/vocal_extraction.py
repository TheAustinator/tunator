import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from pydub import AudioSegment

import librosa.display
# rate, data = wf.read('clairvoyant.wav')


def main():
    orig_filepath = "music/the_story_so_far/The Story So Far 'Small Talk'-LHIVOa-9AgE.wav"
    sides_subtracted = 'sides_subtracted.wav'
    vocals_isolated = 'vocals_isolated.wav'
    vocals_isolated_phased = 'vocals_isolated_phased.wav'
    working_dir = 'working_dir'

    sound_stereo = AudioSegment.from_file(orig_filepath, format='wav')
    os.chdir(working_dir)

    sound_L = sound_stereo.split_to_mono()[0]
    sound_R = sound_stereo.split_to_mono()[1]
    sound_S = sound_L + sound_R.invert_phase()
    sound_voice = sound_stereo + sound_S.invert_phase()
    sound_voice.export(sides_subtracted, format='wav')

    y, sr = librosa.load(sides_subtracted)
    S_full, phase = librosa.magphase(librosa.stft(y))

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max), y_axis='log',
                             x_axis='time', sr=sr)

    plt.colorbar()
    plt.tight_layout()

    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max), y_axis='log',
                             sr=sr)
    plt.title('Full Spectrum')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_background, ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    sf.write(vocals_isolated, librosa.istft(S_foreground), sr, subtype='PCM_24')
    D_foreground = S_foreground * phase
    sf.write(vocals_isolated_phased, librosa.istft(D_foreground), sr, subtype='PCM_24')


if __name__ == '__main__':
    main()
