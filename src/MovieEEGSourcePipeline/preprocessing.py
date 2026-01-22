import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


def load_eeg(sub_id, movie_order):
    """Load raw EEG data for a given subject and movie order."""
    file_name = f'sub{sub_id}_{movie_order}'
    eeg_path = f'data/eeg/{file_name}.edf'
    raw = mne.io.read_raw_edf(eeg_path, preload=True)
    return raw, file_name

def crop_offset(raw, file_name):
    """Crop the raw data based on film start offsets."""
    offset = pd.read_csv('data/film_start.csv')
    idx = file_name + '.bdf'
    start_time = offset.set_index('file').loc[idx, 'film_start']

    # sanity check before cutting the offset 
    status_ch_events = mne.find_events(raw, stim_channel="Status")

    if status_ch_events[1, 0] == start_time * raw.info['sfreq']:  # the timing of the second event in the satus channel should corresponds to the movie start (where we cut the data)
        print('The offset is compatible with Status channel')
        raw.crop(tmin=start_time)
    else:
        ValueError('!!!The offset is NOT compatible with Status channel... DID NOT CROP!!!')
    
    return raw

def set_montage(raw, montage='standard_1020'):
    mon = mne.channels.make_standard_montage(montage)
    raw.set_montage(mon)

def mark_bads(raw, bads):
    if isinstance(bads, float) and np.isnan(bads):
        pass  # no bad channels

    elif isinstance(bads, str) and bads.startswith('['):
        # stringified list: "['P7', 'T8']"
        raw.info['bads'] = re.findall(r"'([^']+)'", bads)

    elif isinstance(bads, str):
        # single channel: 'O1'
        raw.info['bads'] = [bads]

    else:
        raise ValueError(f"Unexpected bad_channels entry: {bads}")

def run_ica(raw, file_name, detect_muscle_ics=False, report=True):
    # Vertical EOG proxy: Fp1 - Cz
    raw = mne.set_bipolar_reference(raw, "Fp1", "Cz", ch_name="VEOG", drop_refs=False)
    raw.set_channel_types({"VEOG": "eog"})

    # Horizontal EOG proxy (optional): F7 - F8
    raw = mne.set_bipolar_reference(raw, "F7", "F8", ch_name="HEOG", drop_refs=False)
    raw.set_channel_types({"HEOG": "eog"})

    # lowpass filtered data for ICA fitting
    raw_filt = raw.copy().filter(None, 40., picks=["eeg", "eog"])
    ica = mne.preprocessing.ICA(n_components=0.99, method="fastica", random_state=97)
    ica.fit(raw_filt, picks="eeg", reject_by_annotation=True)

    eog_inds_v, scores_v = ica.find_bads_eog(raw, ch_name="VEOG")
    eog_inds_h, scores_h = ica.find_bads_eog(raw, ch_name="HEOG")
    muscle_inds = []
    if detect_muscle_ics:
        muscle_inds, muscle_scores = ica.find_bads_muscle(raw)
        muscle_inds = [i for i in muscle_inds if abs(muscle_scores[i]) > 0.9]

    eog_inds_v = [i for i in eog_inds_v if abs(scores_v[i]) >= 0.5]
    eog_inds_h = [i for i in eog_inds_h if abs(scores_h[i]) >= 0.5]

    bad_ic = sorted(set(eog_inds_v + eog_inds_h + muscle_inds))
    ica.exclude = bad_ic

    raw_ica = ica.apply(raw.copy())
    raw_ica.drop_channels(['VEOG', 'HEOG'])

    if report:
        if report:
            # add table and a bar plot of scores to the report
            df_scores = pd.DataFrame({
                "IC": np.arange(ica.n_components_),
                "EOG_V_score": scores_v,
                "EOG_H_score": scores_h,
            })
            if len(muscle_scores):
                df_scores["Muscle_score"] = muscle_scores

            _create_ica_report(raw_filt, ica, df_scores, file_name)

    return raw_ica

def _create_ica_report(raw, ica, df_scores, file_name):
    report = mne.Report(title=f"ICA report – {file_name}")

    report.add_ica(
        ica=ica,
        title="ICA components",
        inst=raw,
        picks=ica.exclude if ica.exclude else None,
        n_jobs=1
    )

    report.add_figure(
        ica.plot_components(show=False),
        title="All IC topographies"
    )

    report.add_figure(
        ica.plot_sources(raw, show=False),
        title="IC time courses"
    )

    html = df_scores.to_html(index=False, float_format="%.3f")

    report.add_html(
        html,
        title="ICA EOG and Muscle correlation scores"
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df_scores["IC"], df_scores["EOG_V_score"], label="VEOG")
    ax.bar(df_scores["IC"], df_scores["EOG_H_score"], alpha=0.6, label="HEOG")
    if hasattr(df_scores, "Muscle_score"):
        ax.bar(df_scores["IC"], df_scores["Muscle_score"], alpha=0.6, label="Muscle")
        ax.axhline(0.9, color="b", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="r", linestyle="--", linewidth=1)
    ax.axhline(-0.5, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("IC")
    ax.set_ylabel("Correlation")
    ax.legend()
    report.add_figure(fig, title="EOG–IC correlation scores")

    fig, ax = plt.subplots(1, len(df_scores.columns)-1, figsize=(20, 4))
    ax[0].hist(df_scores["EOG_V_score"], label="VEOG")
    ax[0].set_title("VEGO Scores")
    ax[1].hist(df_scores["EOG_H_score"], label="HEOG")
    ax[1].set_title("HEGO Scores")
    if hasattr(df_scores, "Muscle_score"):
        ax[2].hist(df_scores["Muscle_score"], label="Muscle")
        ax[2].set_title("Muscle Scores")

    report.add_figure(fig, title="Distribution of Scores")

    report.save(f"data/reports/{file_name}_ica_report.html", overwrite=True)

def epoch_and_reject(cut_dirs_path, raw):
    cuts_art_l = pd.read_csv(cut_dirs_path)  # epoch the data based on the cuts
    cuts_art_l = cuts_art_l[cuts_art_l['Time'] >= 0]  # Drop sentinel (-1), keep real cuts (including 0)
    if cut_dirs_path.stem.__contains__('city_nl'):
        cuts_art_l = cuts_art_l.drop_duplicates('Time')

    events = []
    for _, row in cuts_art_l.iterrows():
        sample = row['Time']
        events.append([int(sample * raw.info['sfreq']+ raw.first_samp), 0, 1])  # first sample is in fact equal to start_time * sampling_rate
    events = np.array(events, dtype=int)

    reject = dict(eeg=150e-6)  # 150 µV in Volts
    epochs_clean = mne.Epochs(raw,
                        events,
                        event_id={'cuts': 1},
                        tmin=-0.2,
                        tmax=1,
                        baseline=None, # no baseline correction for now
                        preload=True,
                        reject=reject,
                        reject_by_annotation=True)

    return epochs_clean

def _calculate_line_ratio(raw):
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(picks="eeg"), sfreq=512, fmin=1, fmax=60, n_fft=8192, verbose=False
    )
    line_ratio = psd[..., (freqs > 49.5) & (freqs < 50.5)].mean() - psd[..., ((freqs > 48) & (freqs < 49)) | ((freqs > 51) & (freqs < 52))].mean()
    return float(line_ratio)
