import music21
import ipdb
import numpy as np
from itertools import islice
from math import floor
from collections import namedtuple
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# from ..utils.peek import peek


def main():
    dir_ = '../music/mini_classical_violin'
    filepath_list = [os.path.join(dir_, fname) for fname in os.listdir(dir_)]
    # for file in filepath_list:
    #     print_parts_countour(open_midi(file).measures(0, 6))
    tensor_gen = gen_batch_tensor(filepath_list, batch_size=5)
    X_batch, Y_batch = next(tensor_gen)
    print(X_batch.shape, Y_batch.shape)


def open_midi(path, remove_drums=True):
    mf = music21.midi.MidiFile()
    mf.open(path)
    mf.read()
    mf.close()
    if remove_drums:
        for i in range(len(mf.tracks)):
            if 10 in mf.tracks[0].getChannels():
                mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    score = music21.midi.translate.midiFileToStream(mf)
    return score


def gen_song_metadata(score_list):
    # TODO: get bpm, key,
    pass


# TODO: for conv, separate instruments with:
# parts = instrument.partitionByInstrument(midi)
# if parts: # file has instrument parts
#     notes_to_parse = parts.parts[0].recurse()
# else: # file has notes in a flat structure
#     notes_to_parse = midi.flat.notes

def gen_batch_tensor(file_list, batch_size, sample_range=(500, 1000)):
    """
    Generates 2-dim tensor for song. The song is normalized to to it's major
    key, with features for the degrees 1 - 12 within the key, and additional
    features for octave and duration.

    Args:
        score_gen (generator):
        sample_range (tuple[int]):

    Yields:
        tensor (np.ndarray): 2-dim input tensor for song. It is helpful to
            imagine it as a 3-dim tensor, but in reality, the "part" dimension
            is flattened into the feature dimension. The dimensions will be
            described below as if it were 3-dim.
            Dimensions:
                time: descretized by smallest time interval in song
                part: melody and chords
                features:
                    0-11: binary encodings for each degree in the major key
                      12: octave (normalized assuming 7 total octaves)
                      13: duration in quarter notes

    """
    _score_gen = (open_midi(path, remove_drums=True).flat for path in file_list)

    # create dict of key relative degrees (1 - 7)

    def _gen_score_tensor():
        def _build_scale(_score):
            key = _score.analyze('key')
            if key.type == 'major':
                pitch_list = [1, 3, 5, 6, 8, 10, 12]
                _scale = {pitch.name: pitch_list[i]
                          for i, pitch in enumerate(key.pitches[0:7])}
            elif key.type == 'minor':
                # convert to major key
                pitch_list = [10, 12, 1, 3, 5, 6, 8]
                _scale = {pitch.name: pitch_list[i]
                          for i, pitch in enumerate(key.pitches[0:7])}
            else:
                raise ValueError()

            scale_lookup = {v: k for k, v in _scale.items()}
            for i in (set(range(1, 13)) - set(pitch_list)):
                note_below = music21.note.Note(scale_lookup[i - 1])
                name = note_below.transpose(1).name
                _scale.update({name: i})

            # update synonymous sharps and flats
            sharps = ('A#', 'C#', 'D#', 'F#', 'G#')
            flats = ('B-', 'D-', 'E-', 'G-', 'A-')
            synonyms = {sharps[i]: flats[i] for i in range(len(sharps))}
            synonyms.update({flats[i]: sharps[i] for i in range(len(flats))})
            _scale.update({k: _scale[synonyms[k]]
                           for k in synonyms if k not in _scale})
            return _scale

        def _update_vector_octave(_time, _event, _type):
            # get row of tone vector
            if _type == 'melody':
                vector_row = 0
            elif _type == 'chord':
                vector_row = 1
            else:
                raise ValueError()

            octave = _event.octave
            octave_norm = octave / 7
            octave_feature_idx = 12
            song_dict[_time][vector_row][octave_feature_idx] = octave_norm

        def _update_vector_note(_time, _event, _type):
            # get row of tone vector
            if _type == 'melody':
                vector_row = 0
            elif _type == 'chord':
                vector_row = 1
            else:
                raise ValueError()
            try:
                tone = scale[_event.name]
            except:
                ipdb.set_trace()
            tone_idx = tone - 1
            song_dict[_time][vector_row][tone_idx] = 1

        def _update_vector_duration(_time, _event, _type):
            # get row of tone vector
            if _type == 'melody':
                vector_row = 0
            elif _type == 'chord':
                vector_row = 1
            else:
                raise ValueError()

            duration = _event.duration._qtrLength
            duration_feature_idx = 13
            song_dict[_time][vector_row][duration_feature_idx] = duration

        def _fill_missing_vectors():
            """
            Adds blank vectors to tensor for timesteps with no notes so that time
            integrity is maintained in output tensor
            """
            time_list = sorted(song_dict)
            end_time = max(time_list)
            min_space = min([j - i for i, j in zip(time_list[:-1], time_list[1:])])
            expected_times = np.array(range(int(end_time / min_space))) * min_space
            missing_times = set(expected_times) - set(time_list)
            if missing_times:
                song_dict.update({time: tone_vector_init for time in missing_times})

        def _slice_tensors_in_time(_tensor):
            """
            Slices tensors to timesteps of song specified by sample_range. Y is one
            timestep ahead of X, but both tensors are the same size.
            Args:
                _tensor (np.ndarray):

            Returns:
                _X (np.ndarray):
                _Y (np.ndarray):
            """
            X_start = sample_range[0]
            X_end = sample_range[1]
            Y_start = sample_range[0] + 1
            Y_end = sample_range[1] + 1
            len_ = sample_range[1] - sample_range[0]
            if len(_tensor) >= Y_end:
                _X = _tensor[X_start:X_end]
                _Y = _tensor[Y_start:Y_end]
            elif X_start < len(_tensor) < Y_start:
                _X = _tensor[-len_ - 1:-1]
                _Y = _tensor[-len_:]
            else:
                raise ValueError('song too short')
            return _X, _Y

        for score_i in range(batch_size):
            score = next(_score_gen)
            scale = _build_scale(score)
            # 12 tones in octave + octave number + duration
            tone_vector_init = np.zeros((2, 14))
            song_dict = dict()

            for event in score.notesAndRests:
                if len(song_dict) >= sample_range[1]:
                    break

                time = event.offset
                if time not in song_dict:
                    song_dict[time] = tone_vector_init

                if isinstance(event, music21.note.Note):
                    _update_vector_octave(time, event, 'melody')
                    _update_vector_duration(time, event, 'melody')
                    _update_vector_note(time, event, 'melody')

                elif isinstance(event, music21.chord.Chord):
                    low_note = event[0]
                    _update_vector_octave(time, low_note, 'chord')
                    _update_vector_duration(time, event, 'chord')
                    for note in event:
                        _update_vector_note(time, note, 'chord')

                elif isinstance(event, music21.note.Rest):
                    pass

                else:
                    raise ValueError()

            _fill_missing_vectors()
            tensor = np.array([song_dict[t].flatten() for t in sorted(song_dict)])
            X, Y = _slice_tensors_in_time(tensor)

            yield X, Y

    n_batches = floor(len(file_list) / batch_size)
    for batch_i in range(n_batches):

        # initialize 3D tensors
        X_batch = None
        Y_batch = None

        # generate tensor for each score in batch and append to batch tensor
        for score_i in range(batch_size):
            X, Y = next(_gen_score_tensor())
            if X_batch is None:

                # expand X and Y to the "score" dimension so all scores in the
                # batch can be stacked on axis 0
                X_batch = np.expand_dims(X, axis=0)
                Y_batch = np.expand_dims(Y, axis=0)
            else:
                X_expanded = np.expand_dims(X, axis=0)
                Y_expanded = np.expand_dims(Y, axis=0)

                X_batch = np.concatenate((X_batch, X_expanded))
                Y_batch = np.concatenate((Y_batch, Y_expanded))

            # transpose Y to use as time series outputs
        Y_batch = np.swapaxes(Y_batch, 0, 1)

        yield X_batch, Y_batch

        # for i in range(len(X_batch)):
        #     X = X_batch[i]
        #     Y = Y_batch[i]
        #     yield X, Y


def gen_midi(tone_tensor):
    pass


def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:
        if isinstance(nt, music21.note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, music21.chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)

    return ret, parent_element


def print_parts_countour(midi):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minPitch = music21.pitch.Pitch('C10').ps
    maxPitch = 0
    xMax = 0

    # Drawing notes.
    for i in range(len(midi.parts)):
        top = midi.parts[i].flat.notes
        y, parent_element = extract_notes(top)
        if (len(y) < 1): continue

        x = [n.offset for n in parent_element]
        ax.scatter(x, y, alpha=0.6, s=7)

        aux = min(y)
        if (aux < minPitch): minPitch = aux

        aux = max(y)
        if (aux > maxPitch): maxPitch = aux

        aux = max(x)
        if (aux > xMax): xMax = aux

    for i in range(1, 10):
        linePitch = music21.pitch.Pitch('C{0}'.format(i)).ps
        if (linePitch > minPitch and linePitch < maxPitch):
            ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))

    plt.ylabel("Note index (each octave has 12 notes)")
    plt.xlabel("Number of quarter notes (beats)")
    plt.title('')
    plt.show()


if __name__ == '__main__':
    main()
