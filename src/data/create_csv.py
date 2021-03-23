import os
import click
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """Create CSV file with dataset informations"""

    project_dir = Path(__file__).resolve().parents[2]

    my_classes = ['anger', 'fear', 'happyness', 'neutral',
                'sadness', 'surprise', 'tense']

    map_class_to_id = {'anger': 0,
                    'fear': 1,
                    'happyness': 2,
                    'neutral': 3,
                    'sadness': 4,
                    'surprise': 5,
                    'tense': 6}

    relative_path = Path('data/raw/emotion_portuguese_database')
    ds_path = os.path.join(project_dir, relative_path)
    data = list()

    for subdir, dirs, files in os.walk(ds_path):
        for filename in files:
            nm, ext = os.path.splitext(filename)
            if ext.lower().endswith('.wav'):
                file_fullpath = os.path.join(subdir, filename)
                label = subdir.split('/')[-1]
                data.append((file_fullpath, label))

    df = pd.DataFrame(data, columns=['file_fullpath', 'label'])

    X = df['file_fullpath']
    y = df['label']

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

    tmp_train_serie = pd.Series(1, index=X_train.index, name='fold')
    tmp_valid_serie = pd.Series(4, index=X_valid.index, name='fold')
    tmp_test_serie = pd.Series(5, index=X_test.index, name='fold')

    df_folds = pd.concat([tmp_train_serie, tmp_valid_serie, tmp_test_serie])

    df_final = pd.concat([df, df_folds], axis=1)

    class_id = df_final['label'].apply(lambda name: map_class_to_id[name])
    df_final = df_final.assign(target=class_id)

    save_path = os.path.join(input_filepath, 'pt_database.csv')
    df_final.to_csv(save_path)

# 'data/processed/emotion_portuguese_database/pt_database.csv'

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files

    main()