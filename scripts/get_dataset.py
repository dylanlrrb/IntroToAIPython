from subprocess import call
from  os import path
import constants

def verify_and_download ():
  with open(f'{constants.WORKING_DIR}/datasets.txt', 'r') as file:
    for line in file.readlines():
      DATASET_URL, UNCOMPRESSED_DATASET_NAME, DATASET_VERSION = line.strip().split(' ')
      print(f'checking if required dataset "{DATASET_VERSION}/{UNCOMPRESSED_DATASET_NAME}" exists.')
      path_already_exists = path.isdir(f'{constants.WORKING_DIR}/datasets/{DATASET_VERSION}/{UNCOMPRESSED_DATASET_NAME}')
      if not path_already_exists:
        print(f'dataset "{DATASET_VERSION}/{UNCOMPRESSED_DATASET_NAME}" does not exist, downloading...')
        call([
          'curl',
          '-o',
          f'{constants.WORKING_DIR}/datasets/{DATASET_VERSION}/{UNCOMPRESSED_DATASET_NAME}.zip',
          DATASET_URL,
          '--create-dirs'
        ])
        call([
          'unzip',
          f'{constants.WORKING_DIR}/datasets/{DATASET_VERSION}/{UNCOMPRESSED_DATASET_NAME}.zip',
          '-d',
          f'{constants.WORKING_DIR}/datasets/{DATASET_VERSION}'
        ])
        call([
          'rm',
          f'{constants.WORKING_DIR}/datasets/{DATASET_VERSION}/{UNCOMPRESSED_DATASET_NAME}.zip',
          ])
      else:
        print(f'specified dataset "{DATASET_VERSION}/{UNCOMPRESSED_DATASET_NAME}" already exists.')

if __name__ == "__main__":
  verify_and_download()