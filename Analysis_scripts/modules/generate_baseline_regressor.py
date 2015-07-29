# -*- coding: utf-8 -*-
"""
Script to read in a Presentation-logfile and create a "reverse" regressor,
i.e. models the interstimulus interval. 
"""

import pandas as pd
import numpy as np

log_path = 'H:\da01-DynAff_narrative_vs123123.log'
base_dir = os.path.dirname(log_path)
pulsecode = 100

df = pd.read_table(log_path, sep = '\t',skiprows=3,header = 0, skip_blank_lines=True)

# Convert to numeric and drop all rows until first pulse
df = df.convert_objects(convert_numeric = True)
pulse_idx = int((np.where(df['Code'] == pulsecode)[0]))

# pulse_t = absolute time of first pulse
pulse_t = df['Time'][df['Code'] == pulsecode]

df = df.drop(range(pulse_idx))

iso_stimtimes = np.array(df['Time'][df['Code'] < 101])

audio_onset = iso_stimtimes[1:] + 10
audio_onset = np.append(0.0, audio_onset)
audio_onset = np.delete(audio_onset,-1)

audio_dur = np.append(iso_stimtimes[1],np.diff(iso_stimtimes[1:]) - 10)

to_write = pd.DataFrame()
to_write['Time'] = audio_onset
to_write['Duration'] = audio_dur
to_write['Weight'] = np.ones(len(audio_onset))

path_bfsl = '%s/%s.bfsl' % (base_dir,'narrative_audio')
to_write.to_csv(path_bfsl, sep = '\t', header = False, index = False)