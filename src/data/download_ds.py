# -*- coding: utf-8 -*-
from git import Repo
import click
import logging
from pathlib import Path


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
def main(dataset_path):
    """ Download dataset to data/raw/file_name"""

    logger = logging.getLogger(__name__)
    logger.info('Downloading dataset')
    git_url = 'git@github.com:TulioChiodi/emotion_portuguese_database.git'
    Repo.clone_from(git_url, dataset_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
