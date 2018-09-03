import numpy as np
import tensorflow as tf
import os.path
import sys
from music21 import *

VERBOSE = True

SAVE_FILE_NAME = "c:/tmp/RTBach_Model_Savefile.ckpt"

Train = False
Compose = True

## voice index
VoiceNum = 4
Sop, Alto, Tenor, Bass = range(VoiceNum)
voiceNames = ['Soprano', 'Alto', 'Tenor', 'Bass']

## pitch constants
PitchRange = 36  # three octaves
PitchCenters = [72, 67, 55, 48]  # corresponding to [C5, G4, G3, C3]
RestPitch = 0

## duration contants
DurationRange = 17  # whole tone maximum
DurationPerQuarter = 4

## flag constants
FlagRange = 4
End, Start, Normal, Cadence = range(FlagRange)


def get_offset_pitch(normal_pitch, pitch_center):
    return normal_pitch - pitch_center + PitchRange / 2


def get_normal_pitch(offset_pitch, pitch_center):
    return offset_pitch + pitch_center - PitchRange / 2


def get_note_at(location, notes):
    for note in notes:
        if (note.location == location):
            n = Note(note)
            return n

        elif (note.location < location and note.location + note.duration > location):
            duration = note.location + note.duration - location
            n = Note(note, duration=duration)
            return n

    return None


class Note:
    def __init__(self, note=None, location=-1, pitch=-1, duration=-1, flag=-1):
        if (note is not None):
            self.location = note.location
            self.pitch = note.pitch
            self.duration = note.duration
            self.flag = note.flag

        if location != -1: self.location = location
        if pitch != -1: self.pitch = pitch
        if duration != -1: self.duration = duration
        if flag != -1: self.flag = flag

        if (flag == Start):
            self.location = -1
            self.pitch = RestPitch
            self.duration = 0

        if (flag == End):
            self.pitch = RestPitch
            self.duration = 0

    def __str__(self):
        return "location: {} flag: {} duration: {} pitch: {}".format(self.location, self.flag, self.duration,
                                                                     self.pitch)


def getKeyDiff(keySig):
    keyDict = {'c major': 0, 'c# major': -1, 'd- major': -1, 'd major': -2, 'd# major': -3, 'e- major': -3,
               'e major': -4, 'f major': -5, 'f# major': 6, 'g- major': 6, 'g major': 5, 'g# major': 4,
               'a- major': 4, 'a major': 3, 'a# major': 2, 'b- major': 2, 'b major': 1,
               'c minor': -3, 'c# minor': -4, 'd- minor': -4, 'd minor': -5, 'd# minor': 6, 'e- minor': 6,
               'e minor': 5, 'f minor': 4, 'f# minor': 3, 'g- minor': 3, 'g minor': 2, 'g# minor': 1,
               'a- minor': 1, 'a minor': 0, 'a# minor': -1, 'b- minor': 1, 'b minor': -2}

    return keyDict[keySig.lower()]


class Voice:
    def __init__(self, part, pitch_center):
        self.notes = []
        self.notes.append(Note(flag=Start))
        location = 0
        key_diff = 0
        for measure in part.getElementsByClass(stream.Measure):
            keySignature = measure.keySignature
            if (keySignature != None):
                key_diff = getKeyDiff(str(keySignature))

            for n in measure.getElementsByClass(note.GeneralNote):
                fermata = False
                for expression in n.expressions:
                    if isinstance(expression, expressions.Fermata):
                        fermata = True

                if n.isChord:
                    pitch = get_offset_pitch(n.pitches[0].ps, pitch_center) # + key_diff
                elif n.isNote:
                    pitch = get_offset_pitch(n.pitch.ps, pitch_center) # + key_diff
                else:
                    pitch = RestPitch  # case : rest and otherwise

                location = (measure.offset + n.offset) * DurationPerQuarter
                duration = n.duration.quarterLength * DurationPerQuarter
                flag = Cadence if fermata else Normal

                # range validation
                if (pitch >= PitchRange): pitch = PitchRange - 1
                if (duration >= DurationRange): duration = DurationRange - 1

                new_note = Note(pitch=pitch, location=location, duration=duration, flag=flag)
                self.notes.append(new_note)

        self.notes.append(Note(flag=End, location=location + duration))

    def get_corresponding_note(self, pivot_note):
        return get_note_at(pivot_note.location, self.notes)


class Piece:
    def __init__(self, stream):
        self.inited = False
        self.voices = [None] * VoiceNum

        for part in stream.parts:
            for voice_index in range(VoiceNum):
                if (part.id == voiceNames[voice_index]):
                    print(part.id)
                    self.voices[voice_index] = Voice(part, PitchCenters[voice_index])

        # voice initialize check
        for voice in self.voices:
            if (voice is None): return

        self.inited = True

    def get_notes(self, pivotvoice_index, destvoice_index=None):
        if (destvoice_index is None): destvoice_index = pivotvoice_index

        return [self.voices[destvoice_index].get_corresponding_note(note) for note in
                self.voices[pivotvoice_index].notes]


## direction constants
DirectionNum = 2
Forward, Backward = range(DirectionNum)

## notewise propagation lists
voice_index_list = [Sop, Bass, Tenor, Alto]
notewiselists = [None] * VoiceNum

notewiselists[Sop] = []
notewiselists[Bass] = [[Forward, Bass, Sop], [Backward, Bass, Sop]]
notewiselists[Tenor] = [[Forward, Tenor, Sop], [Forward, Tenor, Bass],
                        [Backward, Tenor, Sop], [Backward, Tenor, Bass]]
notewiselists[Alto] = [[Forward, Alto, Sop], [Forward, Alto, Bass], [Forward, Alto, Tenor],
                       [Backward, Alto, Sop], [Backward, Alto, Bass], [Backward, Alto, Tenor]]

notewise_all_list = notewiselists[Sop] + notewiselists[Bass] + notewiselists[Tenor] + notewiselists[Alto]

## define notewise propagation size
notewisePropSize = [[[None, None, None, None] for _ in range(VoiceNum)] for _ in range(DirectionNum)]
for direction in [Forward, Backward]:
    for voice_from in voice_index_list:
        for voice_to in voice_index_list:
            if (direction == Forward):
                notewisePropSize[direction][voice_from][voice_to] = 2
            else:
                if voice_from == Bass:
                    notewisePropSize[direction][voice_from][voice_to] = 8
                elif voice_from == Tenor:
                    notewisePropSize[direction][voice_from][voice_to] = 6
                elif voice_from == Alto:
                    notewisePropSize[direction][voice_from][voice_to] = 4

NoteElementNum = 3
NoteVectorSize = FlagRange + DurationRange + PitchRange
Flag, Duration, Pitch = range(NoteElementNum)
rangelist = [FlagRange, DurationRange, PitchRange]


def getNoteName(pitch, pitchCenter):
    pitch = pitchCenter + pitch - PitchRange / 2
    notenames = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B-', 'B']
    currentNoteOctave = abs(int(pitch / 12 - 1))
    currentNotePitch = notenames[int(pitch % 12)]
    return currentNotePitch + str(currentNoteOctave)


def get_notes_from(location, notelist, prop_size, direction):
    notes = []

    current_note = get_note_at(location, notelist)
    for note in notelist if direction == Forward else reversed(notelist):
        if (direction == Forward and note.location < location) or \
                (direction == Backward and note.location > location):
            notes.append(Note(note))

    if (current_note is not None):
        notes.append(current_note)

    notes = notes[-prop_size:len(notes)]
    while (len(notes) < prop_size):
        notes.insert(0, Note(flag=End))

    return notes


def get_max_item_num(itemslist):
    max_num = 0
    for items in itemslist:
        if (len(items) > max_num):
            max_num = len(items)

    return max_num


NumberOfLayers = 2


class Trainer:
    def __init__(self, batch_size, prop_size, state_size, save_period, save_file_name):
        self.voices = [[[] for _ in range(VoiceNum)] for _ in range(VoiceNum)]
        self.notewise = [[[[] for _ in range(VoiceNum)] for _ in range(VoiceNum)] for _ in range(DirectionNum)]

        # init field constants
        self.batch_size = batch_size
        self.prop_size = prop_size
        self.state_size = state_size
        self.save_file_name = save_file_name
        self.save_period = save_period

        # init RNN Cell
        self.main_cell = tf.contrib.rnn.BasicRNNCell(state_size)
        self.main_cell = tf.contrib.rnn.MultiRNNCell([self.main_cell] * NumberOfLayers, state_is_tuple=True)

        self.notewise_cell = tf.contrib.rnn.BasicRNNCell(state_size)
        self.notewise_cell = tf.contrib.rnn.MultiRNNCell([self.notewise_cell] * NumberOfLayers, state_is_tuple=True)

        self.piece_num = 0

    def input(self, piece):
        temp_voices = [[[] for _ in range(VoiceNum)] for _ in range(VoiceNum)]
        temp_notewise = [[[[] for _ in range(VoiceNum)] for _ in range(VoiceNum)] for _ in range(DirectionNum)]

        # input voicewise notes
        for voice_from in voice_index_list:
            for voice_to in voice_index_list:
                notes = piece.get_notes(voice_from, voice_to)
                temp_voices[voice_from][voice_to].append(notes)

        self.piece_num += 1

        # input notewise propagation notes in each voices
        for [direction, voice_from, voice_to] in notewise_all_list:
            temp_notewise[direction][voice_from][voice_to] = [None] * len(temp_voices[voice_from][voice_to])

            for i in range(len(temp_voices[voice_from][voice_to])):
                temp_notewise[direction][voice_from][voice_to][i] = \
                    [None] * (len(temp_voices[voice_from][voice_to][i]) - 1)

                for j in range(len(temp_voices[voice_from][voice_to][i]) - 1):
                    location = temp_voices[voice_from][voice_from][i][j + 1].location
                    notelist = temp_voices[voice_to][voice_to][i]
                    prop_size = notewisePropSize[direction][voice_from][voice_to]
                    temp_notewise[direction][voice_from][voice_to][i][j] = get_notes_from(location, notelist, prop_size,
                                                                                          direction)

        # append voicewise notes to field voices list
        for voice_from in voice_index_list:
            for voice_to in voice_index_list:
                self.voices[voice_from][voice_to] += temp_voices[voice_from][voice_to]

        # append notewise propagation notes to field notewise list
        for [direction, voice_from, voice_to] in notewise_all_list:
            self.notewise[direction][voice_from][voice_to] += temp_notewise[direction][voice_from][voice_to]

    def get_input(self, piece_num, note_num, roll=0):
        data = np.zeros(shape=(VoiceNum, piece_num, note_num, NoteElementNum), dtype=np.int32)

        for voice_index in range(VoiceNum):
            piecelist = self.voices[voice_index][voice_index]

            for piece_index in range(min(piece_num, len(piecelist))):
                notelist = piecelist[piece_index]

                for note_index in range(min(note_num, len(notelist)) + roll):
                    data[voice_index][piece_index][note_index][Flag] = notelist[note_index - roll].flag
                    data[voice_index][piece_index][note_index][Duration] = notelist[note_index - roll].duration
                    data[voice_index][piece_index][note_index][Pitch] = notelist[note_index - roll].pitch

        return data

    def get_notewise_input(self, piece_num, note_num, notewise_prop_num, notewiselist):
        data = np.zeros(shape=(piece_num, note_num, notewise_prop_num, NoteElementNum), dtype=np.int32)

        for piece_index in range(min(piece_num, len(notewiselist))):
            noteslist = notewiselist[piece_index]

            for notes_index in range(min(note_num, len(noteslist))):
                notes = noteslist[notes_index]

                for note_index in range(min(notewise_prop_num, len(notes))):
                    data[piece_index][notes_index][note_index][Flag] = notes[note_index].flag
                    data[piece_index][notes_index][note_index][Duration] = notes[note_index].duration
                    data[piece_index][notes_index][note_index][Pitch] = notes[note_index].pitch

        return data

    def get_loss(self, inputholder, solutionholder, initstateholdertuple, voice_index, notewiseholders, notewiselist):
        onehot_input = tf.concat(
            [tf.one_hot(inputholder[:, :, Flag], FlagRange),
             tf.one_hot(inputholder[:, :, Duration], DurationRange),
             tf.one_hot(inputholder[:, :, Pitch], PitchRange)], axis=-1)

        with tf.variable_scope(voiceNames[voice_index]) as scope:
            main_states, main_current_state = tf.nn.dynamic_rnn(cell=self.main_cell, inputs=onehot_input,
                                                                initial_state=initstateholdertuple, scope=scope)
        onehot_input = tf.unstack(onehot_input, axis=1)
        main_states = tf.unstack(main_states, axis=1)

        auxstates_lists = []

        forward_auxstates_list = [main_states]
        backward_auxstates_list = [main_states]

        notewise_states = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]
        for direction, voice_from, voice_to in notewiselist:
            notewise_states[direction][voice_from][voice_to] = [None] * self.prop_size

            scopename = "{}{}{}".format(direction, voice_from, voice_to)
            for prop_index in range(self.prop_size):
                with tf.variable_scope(scopename, reuse=False if prop_index == 0 else True) as scope:
                    notewise_input = notewiseholders[direction][voice_from][voice_to][:, prop_index]
                    notewise_onehot_input = tf.concat([tf.one_hot(notewise_input[:, :, Flag], FlagRange),
                                                       tf.one_hot(notewise_input[:, :, Duration], DurationRange),
                                                       tf.one_hot(notewise_input[:, :, Pitch], PitchRange)], axis=-1)
                    _, final_state = tf.nn.dynamic_rnn(cell=self.notewise_cell, inputs=notewise_onehot_input,
                                                       dtype=tf.float32, scope=scope)
                    notewise_states[direction][voice_from][voice_to][prop_index] = final_state[NumberOfLayers - 1]

            if (direction == Forward):
                forward_auxstates_list.append(notewise_states[direction][voice_from][voice_to])
            else:
                backward_auxstates_list.append(notewise_states[direction][voice_from][voice_to])

        auxstates_lists.append(forward_auxstates_list)
        auxstates_lists.append(backward_auxstates_list)

        return self.get_loss_internal(main_states, auxstates_lists, solutionholder, voiceNames[voice_index]), \
               main_current_state

    def get_loss_internal(self, main_states, aux_states_lists, solutionholder, voice_name):
        onehot_solution = [None] * NoteElementNum
        onehot_solution[Flag] = tf.one_hot(solutionholder[:, :, Flag], FlagRange)
        onehot_solution[Duration] = tf.one_hot(solutionholder[:, :, Duration], DurationRange)
        onehot_solution[Pitch] = tf.one_hot(solutionholder[:, :, Pitch], PitchRange)

        loss_total = None
        for aux_index in range(len(aux_states_lists)):
            if (len(aux_states_lists[aux_index]) == 0):
                continue

            aux_states_list = aux_states_lists[aux_index]

            weights = [None] * len(aux_states_list)
            biases = [None] * len(aux_states_list)

            for states_index in range(len(aux_states_list)):
                weight_size = self.state_size
                weights[states_index] = [None] * NoteElementNum
                biases[states_index] = [None] * NoteElementNum

                with tf.variable_scope("{}{}{}".format(aux_index, voice_name, states_index)):
                    for element_index in range(NoteElementNum):
                        weights[states_index][element_index] = tf.get_variable(name="weight" + str(element_index),
                                                                               shape=(
                                                                                   weight_size,
                                                                                   rangelist[element_index]),
                                                                               dtype=tf.float32,
                                                                               initializer=tf.random_normal_initializer())
                        biases[states_index][element_index] = tf.get_variable(name="bias" + str(element_index),
                                                                              shape=(1, rangelist[element_index]),
                                                                              dtype=tf.float32,
                                                                              initializer=tf.constant_initializer(0.0))
                        weight_size += rangelist[element_index]

            output_list = [[] for _ in range(NoteElementNum)]
            logit_list = [[[] for _ in range(self.prop_size)] for _ in range(NoteElementNum)]
            for prop_index in range(self.prop_size):
                output = [None] * NoteElementNum

                for states_index in range(len(aux_states_list)):
                    states_solution_concat = aux_states_list[states_index][prop_index]

                    for element_index in range(NoteElementNum):
                        logit = tf.matmul(states_solution_concat, weights[states_index][element_index]) + \
                                biases[states_index][element_index]

                        logit_list[element_index][prop_index].append(logit)

                        if (output[element_index] is None):
                            output[element_index] = logit
                        else:
                            output[element_index] += logit

                        states_solution_concat = tf.concat(
                            [states_solution_concat, onehot_solution[element_index][:, prop_index]], axis=1)

                for element_index in range(NoteElementNum):
                    output_list[element_index].append(output[element_index])

                    logit_list[element_index][prop_index].append(output[element_index])
                    logit_list[element_index][prop_index] = tf.stack(logit_list[element_index][prop_index], axis=0)

            loss = None
            for element_index in range(NoteElementNum):
                output_list[element_index] = tf.stack(output_list[element_index], axis=1)  # by prop axis
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=solutionholder[:, :, element_index],
                                                                    logits=output_list[element_index])
                if (loss is None):
                    loss = ce
                else:
                    loss += ce

            if (loss_total is None):
                loss_total = loss
            else:
                loss_total += loss

        return loss_total, loss_total

    def train(self):
        # if (self.piece_num % self.batch_size == 0):
        #     piece_aligned_num = self.piece_num
        # else:
        #     piece_aligned_num = int(self.piece_num / self.batch_size) * self.batch_size

        piece_aligned_num = self.piece_num
        self.batch_size = self.piece_num

        print("piece_num: {}, piece_aligned_num: {}, batch_size: {}".format(self.piece_num, piece_aligned_num,
                                                                            self.batch_size))

        max_note_num = np.amax(
            [get_max_item_num(self.voices[voice_from][voice_to]) for voice_from in voice_index_list
             for voice_to in voice_index_list])

        max_note_num = 100

        inputholders = [None] * VoiceNum
        solutionholders = [None] * VoiceNum
        for voice_index in range(VoiceNum):
            inputholders[voice_index] = tf.placeholder(dtype=tf.int32,
                                                       shape=(self.batch_size, self.prop_size, NoteElementNum))
            solutionholders[voice_index] = tf.placeholder(dtype=tf.int32,
                                                          shape=(self.batch_size, self.prop_size, NoteElementNum))
        initstateholder = tf.placeholder(dtype=tf.float32, shape=(NumberOfLayers, self.batch_size, self.state_size))
        initstateholdertuple = tf.unstack(initstateholder, axis=0)
        initstateholdertuple = tuple(initstateholdertuple)

        notewiseholders = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]
        for [direction, voice_from, voice_to] in notewise_all_list:
            notewise_prop_size = notewisePropSize[direction][voice_from][voice_to]
            notewiseholders[direction][voice_from][voice_to] = tf.placeholder(dtype=tf.int32, shape=(
                self.batch_size, self.prop_size, notewise_prop_size, NoteElementNum))

        loss = [None] * VoiceNum
        output = [None] * VoiceNum
        current_state = [None] * VoiceNum
        for voice_index in voice_index_list:
            [loss[voice_index], output[voice_index]], current_state[voice_index] = self.get_loss(
                inputholders[voice_index],
                solutionholders[voice_index], initstateholdertuple,
                voice_index,
                notewiseholders, notewiselists[voice_index])

        context = {}
        context['loss'] = loss
        context['output'] = output
        context['pieceNum'] = piece_aligned_num
        context['noteNum'] = max_note_num
        context['inputholders'] = inputholders
        context['solutionholders'] = solutionholders
        context['initstateholder'] = initstateholder
        context['notewiseholders'] = notewiseholders
        context['currentstate'] = current_state
        self.optimize(context)

    def optimize(self, context):
        if (VERBOSE): print("start optimize")

        if (VERBOSE): print("get inputs")
        # shape = (VoiceNum, piece_aligned_num, max_note_num, NoteElementNum)
        _inputs = self.get_input(context['pieceNum'], context['noteNum'])
        _solutions = self.get_input(context['pieceNum'], context['noteNum'], roll=-1)

        if (VERBOSE): print("get notewise inputs")
        _notewise_inputs = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]
        for direction, voice_from, voice_to in notewise_all_list:
            _notewise_inputs[direction][voice_from][voice_to] = \
                self.get_notewise_input(context['pieceNum'], context['noteNum'],
                                        notewisePropSize[direction][voice_from][voice_to],
                                        self.notewise[direction][voice_from][voice_to])

        train_step = [None] * VoiceNum
        for voice_index in voice_index_list:
            train_step[voice_index] = tf.train.AdadeltaOptimizer(0.5).minimize(context['loss'][voice_index])

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # restore variables from disk
            if (os.path.isfile(self.save_file_name + ".index")):
                saver.restore(sess, self.save_file_name)
                print("variables restored.")
            else:
                sess.run(tf.global_variables_initializer())
                print("variables initialized.")

            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            for k, v in zip(variables_names, values):
                print(k, v)

            fetches = [None] * VoiceNum
            feed = [None] * VoiceNum
            for voice_index in voice_index_list:
                fetches[voice_index] = [train_step[voice_index], context['loss'][voice_index],
                                        context['currentstate'][voice_index], context['output'][voice_index]]

            if (VERBOSE): print("start training...")
            epoch = 0
            while (True):
                epoch += 1
                losslists = [[] for _ in range(VoiceNum)]
                for batch_index in range(int(context['pieceNum'] / self.batch_size)):
                    start_batch_index = batch_index * self.batch_size
                    end_batch_index = start_batch_index + self.batch_size

                    _init_states = [None] * VoiceNum
                    for voice_index in voice_index_list:
                        _init_states[voice_index] = np.zeros(shape=(NumberOfLayers, self.batch_size, self.state_size),
                                                             dtype=np.float32)

                    for prop_index in range(int(context['noteNum'] / self.prop_size)):
                        start_prop_index = prop_index * self.prop_size
                        end_prop_index = start_prop_index + self.prop_size

                        _losses = [None] * VoiceNum
                        _output = [None] * VoiceNum
                        for voice_index in voice_index_list:
                            feed[voice_index] = {context['inputholders'][voice_index]:
                                                     _inputs[voice_index, start_batch_index:end_batch_index,
                                                     start_prop_index:end_prop_index],
                                                 context['solutionholders'][voice_index]:
                                                     _solutions[voice_index, start_batch_index:end_batch_index,
                                                     start_prop_index:end_prop_index],
                                                 context['initstateholder']: _init_states[voice_index]}

                            for direction, voice_from, voice_to in notewiselists[voice_index]:
                                feed[voice_index][context['notewiseholders'][direction][voice_from][voice_to]] = \
                                    _notewise_inputs[direction][voice_from][voice_to][
                                    start_batch_index:end_batch_index, start_prop_index:end_prop_index]

                            _, _losses[voice_index], _init_states[voice_index], _output[voice_index] = \
                                sess.run(fetches=fetches[voice_index],
                                         feed_dict=feed[voice_index])

                            losslists[voice_index].append(_losses[voice_index])

                            # print(_output[Alto])

                loss_mean = [None] * VoiceNum
                for voice_index in voice_index_list:
                    loss_mean[voice_index] = np.mean(losslists[voice_index])

                print("epoch : {} loss : {} ".format(epoch, loss_mean))

                if (epoch % self.save_period == 0):
                    save_path = saver.save(sess, self.save_file_name)
                    print("variables saved in file : {}".format(save_path))

    def get_current_state(self, inputholder, mainstateholdertuple, voice_index):
        onehot_input = tf.concat(
            [tf.one_hot(inputholder[:, Flag], FlagRange),
             tf.one_hot(inputholder[:, Duration], DurationRange),
             tf.one_hot(inputholder[:, Pitch], PitchRange)], axis=-1)

        with tf.variable_scope(voiceNames[voice_index] + '/multi_rnn_cell') as scope:
            s, current_state = self.main_cell(onehot_input, mainstateholdertuple, scope=scope)

        return current_state

    def get_element(self, voice_name, element_index, main_states, auxstates_list, auxholders, auxholdersizes, ratio):
        onehot_aux = None
        for aux_index in range(len(auxholders)):
            temp_onehot = tf.one_hot(auxholders[aux_index], auxholdersizes[aux_index])
            if (onehot_aux is None):
                onehot_aux = temp_onehot
            else:
                onehot_aux = tf.concat([onehot_aux, temp_onehot], axis=-1)

        prob = None
        for aux_index in range(len(auxstates_list)):
            if (len(auxstates_list[aux_index]) == 0):
                continue

            auxstates = auxstates_list[aux_index]

            weights = [None] * len(auxstates)
            biases = [None] * len(auxstates)

            weight_size = self.state_size
            for index in range(element_index):
                weight_size += rangelist[index]

            output = None

            for state_index in range(len(auxstates)):
                with tf.variable_scope("{}{}{}".format(aux_index, voice_name, state_index)):
                    weights[state_index] = tf.get_variable(name="weight" + str(element_index),
                                                           shape=(
                                                               weight_size, rangelist[element_index]),
                                                           dtype=tf.float32,
                                                           initializer=tf.random_normal_initializer())
                    biases[state_index] = tf.get_variable(name="bias" + str(element_index),
                                                          shape=(1, rangelist[element_index]),
                                                          dtype=tf.float32,
                                                          initializer=tf.constant_initializer(0.0))

                    if (onehot_aux is None):
                        state_aux_concat = auxstates[state_index]
                    else:
                        state_aux_concat = tf.concat([auxstates[state_index], onehot_aux], axis=-1)

                    logit = tf.matmul(state_aux_concat, weights[state_index]) + biases[state_index]
                    if (output is None):
                        output = logit
                    else:
                        output += logit

            output = tf.nn.softmax(output)

            if (prob is None):
                prob = output
            else:
                prob = np.power(prob, ratio) * output

        prob = np.power(prob, 1 / len(auxstates_list))
        proper_prob = tf.transpose(tf.transpose(prob) / tf.reduce_sum(prob, axis=-1))
        return proper_prob, prob

    def get_notewise_state(self, notewise_holder, direction, voice_from, voice_to):
        onehot_input = tf.concat(
            [tf.one_hot(notewise_holder[:, Flag], FlagRange),
             tf.one_hot(notewise_holder[:, Duration], DurationRange),
             tf.one_hot(notewise_holder[:, Pitch], PitchRange)], axis=-1)

        scopename = "{}{}{}".format(direction, voice_from, voice_to)
        with tf.variable_scope(scopename) as scope:
            _, notewise_state = tf.nn.dynamic_rnn(self.notewise_cell, tf.expand_dims(onehot_input, axis=0),
                                                  dtype=tf.float32, scope=scope)

        return notewise_state[NumberOfLayers - 1]

    def compose(self, compose_batch_size):
        self.compose_batch_size = compose_batch_size

        ## define tensors
        inputholder = tf.placeholder(dtype=tf.int32, shape=(compose_batch_size, NoteElementNum))
        mainstateholder = tf.placeholder(dtype=tf.float32, shape=(NumberOfLayers, compose_batch_size, self.state_size))
        mainstateholdertuple = tf.unstack(mainstateholder, axis=0)
        mainstateholdertuple = tuple(mainstateholdertuple)

        notewise_stateholders = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]
        for direction, voice_from, voice_to in notewise_all_list:
            notewise_stateholders[direction][voice_from][voice_to] = \
                tf.placeholder(dtype=tf.float32, shape=(compose_batch_size, self.state_size))

        current_states = [None] * VoiceNum
        for voice_index in voice_index_list:
            current_states[voice_index] = self.get_current_state(inputholder, mainstateholdertuple, voice_index)

        notewise_holders = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]
        notewise_states = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]

        for voice_index in voice_index_list:
            for direction, voice_from, voice_to in notewiselists[voice_index]:
                notewise_holders[direction][voice_from][voice_to] = tf.placeholder(dtype=tf.int32, shape=(
                    notewisePropSize[direction][voice_from][voice_to], NoteElementNum))
                notewise_states[direction][voice_from][voice_to] = \
                    self.get_notewise_state(notewise_holders[direction][voice_from][voice_to], direction, voice_from,
                                            voice_to)

        flagholder = tf.placeholder(dtype=tf.int32, shape=(compose_batch_size))
        durationholder = tf.placeholder(dtype=tf.int32, shape=(compose_batch_size))

        auxstateslist = [[] for _ in range(VoiceNum)]

        for voice_index in voice_index_list:
            forward_auxstates_list = []
            backward_auxstates_list = [mainstateholder[NumberOfLayers - 1]]

            for direction, voice_from, voice_to in notewiselists[voice_index]:
                if (direction == Forward):
                    forward_auxstates_list.append(notewise_stateholders[direction][voice_from][voice_to])
                else:
                    backward_auxstates_list.append(notewise_stateholders[direction][voice_from][voice_to])

            auxstateslist[voice_index].append(forward_auxstates_list)
            auxstateslist[voice_index].append(backward_auxstates_list)

        flag_probs = [None] * VoiceNum
        pitch_probs = [None] * VoiceNum
        duration_probs = [None] * VoiceNum

        for voice_index in voice_index_list:
            ratio = 1
            # if (voice_index == Bass):
            #     ratio = 0
            # if (voice_index == Tenor):
            #     ratio = 1 / 2
            # elif (voice_index == Alto):
            #     ratio = 1

            flag_probs[voice_index] = self.get_element(voiceNames[voice_index], Flag, mainstateholder,
                                                       auxstateslist[voice_index], [], [], ratio)

            duration_probs[voice_index] = self.get_element(voiceNames[voice_index], Duration, mainstateholder,
                                                           auxstateslist[voice_index],
                                                           [flagholder], [rangelist[Flag]], ratio)

            pitch_probs[voice_index] = self.get_element(voiceNames[voice_index], Pitch, mainstateholder,
                                                        auxstateslist[voice_index], [flagholder, durationholder],
                                                        [rangelist[Flag], rangelist[Duration]], ratio)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # restore variables from disk
            if (os.path.isfile(self.save_file_name + ".index")):
                saver.restore(sess, self.save_file_name)
                print("variables restored.")
            else:
                print("variables can't be initialized.")
                sys.exit(-1)

            makecontext = {}
            makecontext['inputholder'] = inputholder
            makecontext['mainstateholder'] = mainstateholder
            makecontext['notewise_stateholders'] = notewise_stateholders
            makecontext['current_states'] = current_states
            makecontext['notewise_holders'] = notewise_holders
            makecontext['notewise_states'] = notewise_states
            makecontext['flagholder'] = flagholder
            makecontext['durationholder'] = durationholder
            makecontext['flag_probs'] = flag_probs
            makecontext['duration_probs'] = duration_probs
            makecontext['pitch_probs'] = pitch_probs

            for _ in range(10):
                notelists = [None] * VoiceNum

                s = stream.Stream()
                parts = [None for _ in range(VoiceNum)]
                for voice_index in [Sop, Alto, Tenor, Bass]:
                    parts[voice_index] = stream.Part()
                    s.insert(0, parts[voice_index])

                for voice_index in voice_index_list:
                    if (voice_index == Sop):
                        notelists[Sop] = []
                        notelists[Sop].append(Note(flag=Start))
                        notelists[Sop].append(Note(location=0, flag=Normal, pitch=13, duration=3))
                        notelists[Sop].append(Note(location=3, flag=Normal, pitch=13, duration=1))
                        notelists[Sop].append(Note(location=4, flag=Normal, pitch=15, duration=4))
                        notelists[Sop].append(Note(location=8, flag=Normal, pitch=13, duration=4))
                        notelists[Sop].append(Note(location=12, flag=Normal, pitch=18, duration=4))
                        notelists[Sop].append(Note(location=16, flag=Normal, pitch=17, duration=8))

                        notelists[Sop].append(Note(location=24, flag=Normal, pitch=13, duration=3))
                        notelists[Sop].append(Note(location=27, flag=Normal, pitch=13, duration=1))
                        notelists[Sop].append(Note(location=28, flag=Normal, pitch=15, duration=4))
                        notelists[Sop].append(Note(location=32, flag=Normal, pitch=13, duration=4))
                        notelists[Sop].append(Note(location=36, flag=Normal, pitch=20, duration=4))
                        notelists[Sop].append(Note(location=40, flag=Normal, pitch=18, duration=8))

                        notelists[Sop].append(Note(location=48, flag=Normal, pitch=13, duration=3))
                        notelists[Sop].append(Note(location=51, flag=Normal, pitch=13, duration=1))
                        notelists[Sop].append(Note(location=52, flag=Normal, pitch=25, duration=4))
                        notelists[Sop].append(Note(location=56, flag=Normal, pitch=22, duration=4))
                        notelists[Sop].append(Note(location=60, flag=Normal, pitch=18, duration=4))
                        notelists[Sop].append(Note(location=64, flag=Normal, pitch=17, duration=4))
                        notelists[Sop].append(Note(location=68, flag=Cadence, pitch=15, duration=4))

                        notelists[Sop].append(Note(location=72, flag=Normal, pitch=23, duration=3))
                        notelists[Sop].append(Note(location=75, flag=Normal, pitch=23, duration=1))
                        notelists[Sop].append(Note(location=76, flag=Normal, pitch=22, duration=4))
                        notelists[Sop].append(Note(location=80, flag=Normal, pitch=18, duration=4))
                        notelists[Sop].append(Note(location=84, flag=Normal, pitch=20, duration=4))
                        notelists[Sop].append(Note(location=88, flag=Cadence, pitch=18, duration=12))

                        notelists[Sop].append(Note(location=100, flag=End))
                    else:
                        notelists[voice_index] = self.make_melody(sess, voice_index, makecontext, notelists)
                    self.appendNotes(notelists[voice_index], parts[voice_index], PitchCenters[voice_index])

                s.show('musicxml')

    def appendNotes(self, noteList, part, pitchCenter):
        for noteInList in noteList:
            if (noteInList.flag == Start or noteInList.flag == End):
                continue

            if (noteInList.pitch == RestPitch):
                n = note.Rest()
            else:
                n = note.Note()
                noteName = getNoteName(noteInList.pitch, pitchCenter)
                n.pitch.name = noteName

            n.duration.type, n.duration.dots = self.getDurationType(noteInList.duration)
            if (noteInList.flag == Cadence):
                n.expressions.append(expressions.Fermata())

            part.append(n)

    def getDurationType(self, durationIndex):
        if (durationIndex == 8):
            return 'half', 0
        elif (durationIndex == 12):
            return 'half', 1
        elif (durationIndex == 16):
            return 'whole', 0
        elif (durationIndex == 3):
            return 'eighth', 1
        elif (durationIndex == 2):
            return 'eighth', 0
        elif (durationIndex == 1):
            return '16th', 0
        elif (durationIndex == 0):
            return '32nd', 0
        elif (durationIndex == 6):
            return 'quarter', 1
        else:
            return 'quarter', 0

    def notes_to_inputs(self, notes, input_size):
        inputs = np.zeros(shape=(input_size, NoteElementNum), dtype=np.int32)

        for note_index in range(len(notes)):
            inputs[note_index, Flag] = notes[note_index].flag
            inputs[note_index, Duration] = notes[note_index].duration
            inputs[note_index, Pitch] = notes[note_index].pitch

        return inputs

    def make_melody(self, sess, voice_index, context, notelists):
        notelist = []
        notelist.append(Note(flag=Start))
        _prev_state = np.zeros(shape=(NumberOfLayers, self.compose_batch_size, self.state_size))
        _current_input = np.zeros(shape=(self.compose_batch_size, NoteElementNum))
        for i in range(self.compose_batch_size):
            _current_input[i, Flag] = Start

        locations = [0 for _ in range(self.compose_batch_size)]
        _notewise_states = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]
        for direction, voice_from, voice_to in notewiselists[voice_index]:
            _notewise_states[direction][voice_from][voice_to] = {}

        scores = np.ones(shape=(self.compose_batch_size), dtype=np.float64)
        notes_num = [0 for _ in range(self.compose_batch_size)]
        ended = [False for _ in range(self.compose_batch_size)]
        noteslist = [[] for _ in range(self.compose_batch_size)]

        for batch_index in range(self.compose_batch_size):
            noteslist[batch_index].append(Note(flag=Start))

        i = 0
        while (i < 100):
            i += 1
            # print("in while loop {}".format(i))

            _current_state = sess.run(context['current_states'][voice_index],
                                      feed_dict={context['inputholder']: _current_input,
                                                 context['mainstateholder']: _prev_state})

            _batch_notewise_states = [[[None, None, None, None] for _ in range(4)] for _ in range(2)]

            for loc_index in range(self.compose_batch_size):
                for direction, voice_from, voice_to in notewiselists[voice_index]:
                    try:
                        _temp = _notewise_states[direction][voice_from][voice_to][str(locations[loc_index])]
                    except KeyError:
                        notes = get_notes_from(locations[loc_index], notelists[voice_to],
                                               notewisePropSize[direction][voice_from][voice_to], direction)
                        _notewise_inputs = self.notes_to_inputs(notes,
                                                                notewisePropSize[direction][voice_from][voice_to])
                        _temp = _notewise_states[direction][voice_from][voice_to][str(locations[loc_index])] = \
                            sess.run(context['notewise_states'][direction][voice_from][voice_to], feed_dict={
                                context['notewise_holders'][direction][voice_from][voice_to]: _notewise_inputs})

                    if (_batch_notewise_states[direction][voice_from][voice_to] is None):
                        _batch_notewise_states[direction][voice_from][voice_to] = _temp
                    else:
                        _batch_notewise_states[direction][voice_from][voice_to] = \
                            np.concatenate([_batch_notewise_states[direction][voice_from][voice_to], _temp], axis=0)

            feed = {}
            feed[context['mainstateholder']] = _current_state
            feed[context['inputholder']] = _current_input
            for direction, voice_from, voice_to in notewiselists[voice_index]:
                feed[context['notewise_stateholders'][direction][voice_from][voice_to]] = \
                    _batch_notewise_states[direction][voice_from][voice_to]

            DrawNum = [[None for _ in range(VoiceNum)] for _ in range(NoteElementNum)]
            DrawNum[Flag][Sop] = 10
            DrawNum[Flag][Bass] = 100
            DrawNum[Flag][Tenor] = 100
            DrawNum[Flag][Alto] = 100

            DrawNum[Duration][Sop] = 100
            DrawNum[Duration][Bass] = 50
            DrawNum[Duration][Tenor] = 50
            DrawNum[Duration][Alto] = 50

            DrawNum[Pitch][Sop] = 10
            DrawNum[Pitch][Bass] = 10
            DrawNum[Pitch][Tenor] = 10
            DrawNum[Pitch][Alto] = 10

            # get flag
            _flag_probs, _flag_probs_n = sess.run(context['flag_probs'][voice_index], feed_dict=feed)
            _flag_data, flagscore = self.choose(_flag_probs, _flag_probs_n, DrawNum[Flag][voice_index])

            # get duration
            feed[context['flagholder']] = _flag_data
            _duration_probs, _duration_probs_n = sess.run(context['duration_probs'][voice_index], feed_dict=feed)

            _duration_data, durationscore = self.choose(_duration_probs, _duration_probs_n,
                                                        DrawNum[Duration][voice_index])

            # get pitch
            feed[context['durationholder']] = _duration_data
            _pitch_probs, _pitch_probs_n = sess.run(context['pitch_probs'][voice_index], feed_dict=feed)
            _pitch_data, pitchscore = self.choose(_pitch_probs, _pitch_probs_n, DrawNum[Pitch][voice_index])

            _prev_state = _current_state
            _current_input = np.concatenate([np.expand_dims(_flag_data, axis=-1),
                                             np.expand_dims(_duration_data, axis=-1),
                                             np.expand_dims(_pitch_data, axis=-1)], axis=-1)

            all_ended = True

            for batch_index in range(self.compose_batch_size):
                if (ended[batch_index] == False):
                    new_note = Note(location=locations[batch_index], flag=_flag_data[batch_index],
                                    duration=_duration_data[batch_index], pitch=_pitch_data[batch_index])
                    noteslist[batch_index].append(new_note)
                    locations[batch_index] += _duration_data[batch_index]
                    notes_num[batch_index] += 1
                    scores[batch_index] *= pitchscore[batch_index]

                if (ended[batch_index] == False and _flag_data[batch_index] == End):
                    ended[batch_index] = True

                all_ended = all_ended and ended[batch_index]

            # print(ended)
            if (all_ended):
                break

                # print(scores)

        for batch_index in range(self.compose_batch_size):
            scores[batch_index] = np.power(scores[batch_index], 1 / notes_num[batch_index])
            if (not ended[batch_index]):
                scores[batch_index] = 0
            if (notes_num[batch_index] < 10):
                scores[batch_index] = 0

        print(scores)
        print(notes_num)
        return noteslist[np.argmax(scores)]

    def make_proper(self, a):
        ret = a / np.expand_dims(np.sum(a, axis=-1), axis=-1)
        ret *= 0.99
        return ret

    def choose(self, probs, probs_n, draw):
        proper_probs = self.make_proper(probs)
        score = np.zeros(shape=(self.compose_batch_size), dtype=np.float32)
        output = np.zeros(shape=(self.compose_batch_size), dtype=np.int32)

        for i in range(self.compose_batch_size):
            draw_result = np.random.multinomial(draw, proper_probs[i])
            max_index = np.argmax(draw_result)
            output[i] = max_index
            score[i] = probs_n[i, max_index]

        return output, score


trainer = Trainer(batch_size=500, prop_size=8, state_size=64,
                  save_period=100, save_file_name=SAVE_FILE_NAME)

if (Train):
    bach_bundles = corpus.corpora.CoreCorpus().search('bwv')

    i = 0
    for bundle in bach_bundles:
        i += 1
        if (i > 500):
            break

        parsed_stream = bundle.parse()
        piece = Piece(parsed_stream)
        if (not piece.inited): continue
        # parsed_stream.show('musicxml')
        trainer.input(piece)

    trainer.train()

if (Compose):
    trainer.compose(compose_batch_size=1000)