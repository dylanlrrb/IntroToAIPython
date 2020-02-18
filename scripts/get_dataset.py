from subprocess import call
from  os import path
import constants

def verify_and_download ():
  with open(f'{constants.WORKING_DIR}/datasets.txt', 'r') as file:
    for line in file.readlines():
      DATASET_URL, DATASET_DESTINATION = line.strip().split(' ')
      print(f'checking if required dataset "{DATASET_DESTINATION}/" exists.')
      path_already_exists = path.isdir(f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}')
      if not path_already_exists:
        print ('DATASET_URL, DATASET_DESTINATION', DATASET_URL, DATASET_DESTINATION)
        print(f'dataset "{DATASET_DESTINATION}/" does not exist, downloading...')
        print('curl', '-o', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}.zip', DATASET_URL, '--create-dirs')
        call(['curl', '-o', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}.zip', DATASET_URL, '--create-dirs'])
        print('unzip', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}.zip', '-d', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}')
        call(['unzip', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}.zip', '-d', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}'])
        print('rm', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}.zip')
        call(['rm', f'{constants.WORKING_DIR}/datasets/{DATASET_DESTINATION}.zip'])
      else:
        print(f'specified dataset "{DATASET_DESTINATION}/" already exists.')

if __name__ == "__main__":
  verify_and_download()