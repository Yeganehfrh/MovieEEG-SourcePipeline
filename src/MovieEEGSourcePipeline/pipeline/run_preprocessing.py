import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.MovieEEGSourcePipeline.preprocessing import Preprocessing, StatusOffsetError

os.environ["MNE_BROWSER_BACKEND"] = "matplotlib"

logging.basicConfig(
    filename="preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def run_preprocessing_pipeline(cut_path, bad_channels_path, start_time_path, eeg_base_path):
    all_cuts = pd.read_csv(cut_path)
    bad_channels = pd.read_csv(bad_channels_path, index_col=0)
    start_time = pd.read_csv(start_time_path)

    # file namas
    file_names = start_time['file'].values
    file_names = [f.replace('.bdf', '') for f in file_names]
    start_time.set_index('file', inplace=True)

    for file in file_names:
        sub_id, movie, order = file.split("_")
        bad_ch = bad_channels.loc[int(sub_id), f'{movie}_{order}']
        offset = start_time.loc[f'{file}.bdf', 'film_start']
        cuts = all_cuts.groupby('condition').get_group(f"{movie}_{order}").drop('condition', axis=1).reset_index(drop=True)
        eeg_path = Path(f'{eeg_base_path}/{movie}_edf/{movie}_{order}/{file}.edf')
        
        try:
            print(f"Processing subject {sub_id}, movie {movie}...")
            preprocessing = Preprocessing(
                eeg_path=eeg_path,
                start_time=offset,
                bad_channels=bad_ch,
                cuts=cuts,
                detect_muscle_ics=False,
                report=True,
                report_path=Path('data/reports/')
            )
            epochs_clean, line_ratio = preprocessing.run()

            # save epochs
            epochs_clean.save(f'data/epochs/{file}_epo.fif', overwrite=True)

            # log line_ratio
            logger.info(f"Subject {sub_id}, Movie {movie}, Line Noise IC Ratio: {line_ratio}")

            print(f"Subject {sub_id} processed successfully.")
        except StatusOffsetError as e:
            logger.warning(f"Skipping subject {sub_id} due to Status channel issue: {e}")
            print(f"Skipping subject {sub_id} due to Status channel issue. Check log for details.")
        except Exception as e:
            logger.error(f"Error processing subject {sub_id}: {e}")
            print(f"Error processing subject {sub_id}. Check log for details.")

if __name__ == "__main__":
    cuts_csv = 'data/All_conditions_all_cuts.csv'
    bad_channels_csv = 'data/bad_channels_no_ica4muscle.csv'
    start_time_csv = 'data/film_start.csv'
    eeg_base_path = 'data/eeg'
    run_preprocessing_pipeline(cuts_csv,
                               bad_channels_csv,
                               start_time_csv,
                               eeg_base_path)
