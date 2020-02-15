from subprocess import call
from  os import path
import constants

def verify_and_download ():
  path_already_exists = path.isdir(constants.DATASET_DESTINATION)
  if not path_already_exists:
    call(['curl', '-o', f'{constants.DATASET_DESTINATION}.zip', constants.DATASET_URL])
    call(['unzip', f'{constants.DATASET_DESTINATION}.zip', '-d', constants.DATASET_DESTINATION])
    call(['rm', f'{constants.DATASET_DESTINATION}.zip'])
  else:
    print('specified dataset already exists.')

if __name__ == "__main__":
  verify_and_download()