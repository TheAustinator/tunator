from abc import ABC, abstractmethod
from datetime import datetime

import fractions
import glob
from itertools import islice

from itertools import islice
import h5py
import math
import music21 as m21
import numpy as np
import os
import random
import ipdb

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, TimeDistributed, Dropout, CuDNNLSTM, Activation, LSTM
from keras.models import Sequential
from keras.utils import Sequence
from tensorflow.contrib.training import HParams


def main():
    hparams = {
        'learning_rate': 0.001,
        'dropout': 0.2,
        'lstm_units': 512,
        'dense_units': 512,
        'batch_size': 32,
        'timesteps': 256,
        'epochs': 3,
    }
    layers = []
    tunator_lstm = TunatorLSTM()
    tunator_lstm.build_model()
    tunator_lstm.train()
    tunator_lstm.compose(128)
    ipdb.set_trace()


class TunatorLSTM:
    def __init__(self, midi_dir='music/midi/final_fantasy/', hdf5_path='data/songs.hdf5', hparams=None):
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
            'lstm_units': 512,
            'dense_units': 512,
            'batch_size': 32,
            'timesteps': 256,
            'epochs': 3,
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
    def vocab(self):
        return self.tensor_gen.vocab

    @property
    def n_vocab(self):
        return len(self.vocab)

    @property
    def timestamp(self):
        return datetime.now()

    def build_model(self):
        self.model = Sequential()
        input_shape = (None, self.n_vocab)

        self.model.add(CuDNNLSTM(
            self.hparams.lstm_units,
            input_shape=input_shape,
            return_sequences=True,
        ))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(CuDNNLSTM(self.hparams.lstm_units, return_sequences=True))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(CuDNNLSTM(self.hparams.lstm_units, return_sequences=True))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(TimeDistributed(Dense(self.n_vocab)))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(TimeDistributed(Dense(self.n_vocab)))
        self.model.add(Dropout(self.hparams.dropout))

        self.model.add(TimeDistributed(Activation('softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def load_model(self, model_path):
        self.build_model()
        self.model.load_weights(model_path)

    def train(self):
        timestamp = datetime.now()
        log_name = f'note-chord-one-hot-songs_{timestamp}'
        tensorboard = TensorBoard(log_dir=f'logs/{log_name}', histogram_freq=1, write_graph=True, write_grads=True, batch_size=4) #write_images
        # if adding embeddings, add those parameters
        checkpoint_name = 'weights-improvement-epoch_{epoch:02d}-loss_{loss:.4f}.hdf5'
        checkpoint = ModelCheckpoint(
            f'checkpoints/{checkpoint_name}',
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )

        val_slice = list(islice(self.val_tensor_gen, 10))
        X_val_list = list()
        Y_val_list = list()
        for item in val_slice:
            X_val_list.append(item[0])
            Y_val_list.append(item[1])
        X_val = np.concatenate(X_val_list, axis=0)
        Y_val = np.concatenate(Y_val_list, axis=0)
        val_data = (X_val, Y_val)
        self.model.fit_generator(
            self.train_tensor_gen,
            validation_data=val_data,
            # validation_steps=10,
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

    def query_datastore(self, query, grp_path='songs'):
        if os.path.isfile(self.hdf5_path):
            grp = h5py.File(self.hdf5_path, 'r')[grp_path]
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
            midi = m21.converter.parse(file)

            # extract piano, or other
            try:
                midi_parts = m21.instrument.partitionByInstrument(midi).parts
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

                if isinstance(elem, m21.note.Note):
                    notes[time].add(str(elem.pitch))
                elif isinstance(elem, m21.chord.Chord):
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

    def compose(self, timesteps):
        seed_note = None
        while not seed_note:
            with h5py.File(self.hdf5_path) as f:
                grp = f['songs']
                song_names = list(grp.keys())
                song_idx = np.random.randint(0, len(song_names))
                song = grp[song_names[song_idx]]['str_notes']
                note_idx = np.random.randint(0, len(song))
                seed_note = song[note_idx]
        note_to_int = dict((note, i) for i, note in enumerate(self.vocab))
        int_to_note = dict((i, note) for i, note in enumerate(self.vocab))
        note_int = note_to_int[seed_note]
        x = np.zeros(len(note_to_int))
        x[note_int] = 1

        # generate notes
        Y_hat = []
        for note_index in range(timesteps):
            y_hat = self.model.predict(x)
            Y_hat.append(y_hat)
            x = y_hat

        Y_hat_ints = [np.where(vector==1)[0][0] for vector in Y_hat]
        Y_hat_strs = [int_to_note[int_] for int_ in Y_hat_ints]

        self._output_midi(Y_hat_strs)

        return Y_hat_strs

    def _output_midi(self, Y_hat_strs):
        timesteps = len(Y_hat_strs)
        offset = 0
        output_notes = []

        for event_str in Y_hat_strs:    # chord
            if '.' in event_str:
                event_split = event_str.split('.')
                notes = []
                for note_str in event_split:
                    note = m21.note.Note(int(note_str))
                    m21.note.storedInstrument = m21.instrument.Piano()
                    notes.append(note)
                chord = m21.chord.Chord(notes)
                chord.offset = offset
                output_notes.append(chord)
            elif event_str:    # note
                note = m21.note.Note(event_str)
                note.offset = offset
                note.storedInstrument = m21.instrument.Piano()
                output_notes.append(note)
            else:    # rest
                rest = m21.note.Rest()
                rest.offset = offset
                rest.storedInstrument = m21.instrument.Piano()
                output_notes.append(rest)

            offset += .25

        midi = m21.stream.Stream(output_notes)
        midi.write('midi', fp=f'test_output-{timesteps}-{self.timestamp}.mid')


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
