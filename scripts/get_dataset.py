from subprocess import call
from  os import path
import constants

def verify_and_download ():
  with open(f'{constants.WORKING_DIR}/datasets.txt', 'r') as file:
    for line in file.readlines():
      DATASET_URL, DATASET_DESTINATION = line.strip().split(' ')
      path_already_exists = path.isdir(f'{constants.WORKING_DIR}/{DATASET_DESTINATION}')
      if not path_already_exists:
        call(['curl', '-o', f'{constants.WORKING_DIR}/{DATASET_DESTINATION}.zip', DATASET_URL])
        call(['unzip', f'{constants.WORKING_DIR}/{DATASET_DESTINATION}.zip', '-d', constants.WORKING_DIR])
        call(['rm', f'{constants.WORKING_DIR}/{DATASET_DESTINATION}.zip'])
      else:
        print(f'specified dataset "{DATASET_DESTINATION}/" already exists.')

if __name__ == "__main__":
  verify_and_download()