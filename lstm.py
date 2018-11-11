""" This module prepares midi file data and feeds it to the neural
    network for training """
import fractions
import glob
import random
from abc import ABC, abstractmethod
from datetime import datetime

import h5py
import math
import os
import music21
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, CuDNNLSTM, Activation, LSTM
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.contrib.training import HParams
import numpy as np
import ipdb


def main():
    tunator_lstm = TunatorLSTM()
    tunator_lstm.build_network()
    tunator_lstm.train()


class TunatorLSTM:
    def __init__(self, midi_dir='midi_songs/', hdf5_path='data/songs.hdf5', hparams=None):
        self.midi_dir = midi_dir
        self.hdf5_path = hdf5_path
        self._hparams = hparams

        self.song_file_dict = self.get_song_file_dict()
        songs = list(self.song_file_dict)
        random.shuffle(songs)
        split = int(.8 * len(songs))
        self.train_songs = songs[:split]
        self.val_songs = songs[split:]
        self.update_datastore()
        self.tensor_gen = NoteChordOneHotTensorGen(
            songs,
            self.hparams.batch_size,
            self.hparams.timesteps,
            hdf5_path)
        self.train_tensor_gen = NoteChordOneHotTensorGen(
            self.train_songs,
            self.hparams.batch_size,
            self.hparams.timesteps,
            hdf5_path,
            self.tensor_gen.vocab)
        self.val_tensor_gen = NoteChordOneHotTensorGen(
            self.val_songs,
            self.hparams.batch_size,
            self.hparams.timesteps,
            hdf5_path,
            self.tensor_gen.vocab)

    @property
    def hparams(self):
        defaults = {
            'learning_rate': 0.001,
            'dropout': 0.2,
            'lstm_units': 1024,
            'dense_units': 512,
            'batch_size': 16,
            'timesteps': 256,
            'epochs': 10,
        }

        if isinstance(self._hparams, HParams):
            return self._hparams
        elif self._hparams:
            user_entered = self._hparams
        else:
            user_entered = dict()
        combined = dict()
        combined.update(defaults)
        combined.update(user_entered)
        self._hparams = HParams()
        for k, v in combined.items():
            self._hparams.add_hparam(k, v)

        return self._hparams

    @hparams.setter
    def hparams(self, values):
        if not isinstance(self._hparams, HParams):
            self._hparams = HParams()
        for k, v in values.items():
            self._hparams.add_hparam(k, v)

    @property
    def n_vocab(self):
        return len(self.tensor_gen.vocab)

    def build_network(self):
        self.model = Sequential()
        input_shape = (self.hparams.timesteps, self.n_vocab)

        self.model.add(LSTM(
            self.hparams.lstm_units,
            input_shape=input_shape,
            return_sequences=True,
        ))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(LSTM(self.hparams.lstm_units, return_sequences=True))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(LSTM(self.hparams.lstm_units, return_sequences=True))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(TimeDistributed(Dense(self.n_vocab)))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(TimeDistributed(Dense(self.n_vocab)))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(TimeDistributed(Activation('softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def train(self):
        """ train the neural network """
        timestamp = datetime.now()
        log_name = f'note-chord-one-hot-songs_{timestamp}'
        tensorboard = TensorBoard(log_dir=f'logs/{log_name}')
        checkpoint_name = 'weights-improvement-epoch_{epoch:02d}-loss_{loss:.4f}.hdf5'
        checkpoint = ModelCheckpoint(
            f'checkpoints/{checkpoint_name}',
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )

        self.model.fit_generator(
            self.train_tensor_gen,
            validation_data=self.val_tensor_gen,
            steps_per_epoch=self.train_tensor_gen.n_batches,
            epochs=self.hparams.epochs,
            callbacks=[checkpoint, tensorboard]
        )

    def get_song_file_dict(self):
        file_exts = ('*.mid', '*.midi', '*.MID', '*.MIDI')
        song_files = list()
        for ext in file_exts:
            song_files += glob.glob(os.path.join(self.midi_dir, ext))
        get_song_name = lambda x: os.path.splitext(os.path.basename(x))[0]
        song_names = [get_song_name(file) for file in song_files]
        # dict to look up filepath by song name
        song_file_dict = dict(zip(song_names, song_files))
        return song_file_dict

    def query_datastore(self, query, path='songs'):
        if os.path.isfile(self.hdf5_path):
            grp = h5py.File(self.hdf5_path, 'r')[path]
            keys = list(grp.keys())
        else:
            keys = list()

        found = set(query).intersection(keys)
        not_found = set(query) - set(found)
        return found, not_found

    def update_datastore(self):
        """
        Updates HDF5 datastore with note sequences for any songs in midi_dir
        that are not already present in the datastore.
        """
        def _parse_midi(song):
            file = self.song_file_dict[song]
            print(f'parsing {file}...')
            midi = converter.parse(file)

            # extract piano, or otherp
            try:
                midi_parts = instrument.partitionByInstrument(midi).parts
                part = midi_parts[0]
                if not part.partName == 'Piano':
                    pass
                notes_to_parse = part.recurse().notes
                part_i = 0
                while len(notes_to_parse) < 50:
                    part_i += 1
                    part = midi_parts[part_i]
                    notes_to_parse = part.recurse().notes

            except Exception:  # file has notes in a flat structure
                notes_to_parse = midi.flat.chordify().notes

            return notes_to_parse

        def _parse_notes(notes_to_parse):
            notes = dict()
            for elem in notes_to_parse:
                time = elem.offset
                if time not in notes:
                    notes[time] = set()

                if isinstance(elem, note.Note):
                    notes[time].add(str(elem.pitch))
                elif isinstance(elem, chord.Chord):
                    notes[time].update([str(pitch) for pitch in elem.pitches])
                else:
                    raise ValueError()

            # TODO: SongMap slicable hashmap class
            # correct fractional indices
            frac_notes = {k: v for k, v in notes.items() if isinstance(k, fractions.Fraction)}
            for k, v in frac_notes.items():
                del notes[k]
                nearest_quarter = round(k * 4) / 4
                if nearest_quarter in notes:
                    notes[nearest_quarter].update(v)
                else:
                    notes[nearest_quarter] = v

            # fill missing time indices
            time_list = sorted(notes)
            if not time_list:
                raise ValueError()
            end_time = max(time_list)
            min_space = min([j - i for i, j in zip(time_list[:-1], time_list[1:])])
            expected_times = np.array(range(int(end_time / min_space))) * min_space
            missing_times = set(expected_times) - set(time_list)
            if missing_times:
                print(f'filling in {len(missing_times)} missing timepoints in '
                      f'existing {len(notes)}...')
                notes.update({time: set() for time in missing_times})

            # convert notes to a list of strings
            str_notes = ['.'.join(sorted(notes[k])) for k in sorted(notes)]
            # remove leading and trailing rests
            for i in (0, -1):
                while str_notes and str_notes[i] == '':
                    str_notes.pop(i)
            # encoding required by h5py
            str_notes = np.array(str_notes).astype('|S9')

            vocab = np.array(list(set(str_notes))).astype('|S9')
            return str_notes, vocab, min_space

        def _write_to_datastore(str_notes, vocab, min_space):
            with h5py.File(self.hdf5_path, 'a') as f:
                grp = f.create_group(f'songs/{song}')

                grp.create_dataset(
                    name='vocab',
                    shape=(len(vocab), 1),
                    data=vocab,
                    dtype='S9')
                dset_notes = grp.create_dataset(
                    name='str_notes',
                    shape=(len(str_notes), 1),
                    data=str_notes,
                    dtype='S9')
                dset_notes.attrs['spacing'] = min_space

        song_names = set(self.song_file_dict)
        _, missing_songs = self.query_datastore(song_names)

        for song in missing_songs:
            notes_to_parse = _parse_midi(song)
            str_notes, vocab, min_space = _parse_notes(notes_to_parse)
            _write_to_datastore(str_notes, vocab, min_space)


class SongMap:
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass


class TensorGen(ABC):
    def __init__(self, key_list, hdf5_path, batch_size):
        self.key_list = key_list
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.X_list = list()
        self.Y_list = list()

    def __iter__(self):
        while len(self.X_list) < self.batch_size:
            for key in self.key_list:
                X, Y, = self.generate(key)
                yield X, Y

    @abstractmethod
    def generate(self, key):
        pass

    @abstractmethod
    def set_up(self):
        pass

    @abstractmethod
    def tear_down(self):
        pass


class NoteChordOneHotTensorGen(Sequence):
    def __init__(self, songs, batch_size, timesteps, hdf5_path, vocab=None):
        self.songs = songs
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.hdf5_path = hdf5_path

        self.batch_counter = 0
        self.epoch_counter = 0
        self._vocab = vocab
        self.seq_info = self.get_seq_info()
        self.n_batches = math.floor(len(self.seq_info) / self.batch_size)

    def __len__(self):
        return len(self.seq_info)

    def __getitem__(self, idx):
        i = idx % self.n_batches
        batch_info = self.seq_info[i * self.batch_size: (i + 1) * self.batch_size]

        X_list = list()
        Y_list = list()
        with h5py.File(self.hdf5_path, 'r') as f:
            for info in batch_info:
                name = info[0]
                slice_ = info[1]
                grp_song = f[f'songs/{name}/str_notes']
                seq = grp_song[slice_[0]: slice_[1]].flatten()
                X = self.build_vector(seq)
                Y = X[1:]
                X = X[:-1]
                X_list.append(X)
                Y_list.append(Y)

        X_batch = np.array(X_list)
        Y_batch = np.array(Y_list)
        self.batch_counter += 1
        try:
            assert X_batch.shape == Y_batch.shape
            assert X_batch.shape == (self.batch_size, self.timesteps, len(self.vocab))
        except:
            pass
        return X_batch, Y_batch

    @property
    def vocab(self):
        if not self._vocab:
            vocab = set()
            with h5py.File(self.hdf5_path, 'r') as f:
                for song in self.songs:
                    grp = f[f'songs/{song}/vocab']
                    vocab.update(list(grp[:].flat))
            self._vocab = vocab
        return self._vocab

    def get_seq_info(self):
        """
        Builds a lookup dictionary for song samples. The integer keys are
        shuffled, and the values are tuples of the song name and start index
        for the sample. The samples are generated as back-to-back samples from
        each song in `songs` of length `timesteps`, and are looked up in the
        datastore at `hdf5_path`.
        Returns:
            seq_dict (dict): integer keys for randomized sample number and tuple
                values containing song name and sample start index.
                Example:
                    {15: ('Relax - Frankie Goes to Hollywood': 250),
                     47: ('Rethel Bean - Rappin Ronnie': 750),
                     ...
                     }
        """
        seq_info = list()
        with h5py.File(self.hdf5_path, 'r') as f:
            for song in self.songs:
                grp = f[f'songs/{song}/str_notes']
                song_len = len(grp)
                n_seq = math.floor(song_len / (self.timesteps + 1))
                for i in range(n_seq):
                    slice_ = (i * self.timesteps, (i + 1) * self.timesteps + 1)
                    new_seq_info = (song, slice_)
                    seq_info.append(new_seq_info)

        random.shuffle(seq_info)
        return seq_info

    def build_vector(self, seq):
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, i) for i, note in enumerate(self.vocab))

        seq_int = [note_to_int[note] for note in seq]
        X = np.array([np.zeros(len(note_to_int)) for i in seq_int])
        X[np.arange(len(X)), seq_int] = 1
        return X

    def on_epoch_end(self):
        self.epoch_counter += 1


if __name__ == '__main__':
    main()