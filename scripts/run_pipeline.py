import boto3
import constants
from  os import path, remove
import sys
import traceback
from datetime import datetime
from subprocess import call, check_output
from get_dataset import verify_and_download

# This script assumes you have the package 'boto3' installed and accessible from here
s3_client = boto3.client('s3')

def filter_by_dir_name(s3_object):
  if s3_object.key.find(constants.S3_BUCKET_DIR) > -1:
    path = s3_object.key.split('/')
    return (len(path) > 1 and len(path[1]) > 0)
  else:
    return False


def extract_child_dirs(s3_object):
  path = s3_object.key.split('/')
  return path[1]

def try_to_remove(file):
  try:
    remove(f'{constants.WORKING_DIR}/{file}')
  except FileNotFoundError:
    print(f'{constants.WORKING_DIR}/{file} already removed or does not exist')


def tags_to_build ():
  s3 = boto3.resource('s3')
  my_bucket = s3.Bucket(constants.S3_BUCKET)
  
  bucket_contents = [*my_bucket.objects.all()]
  bucket_contents = filter(filter_by_dir_name, bucket_contents)
  bucket_contents = map(extract_child_dirs, bucket_contents)
  bucket_contents = set(bucket_contents)
  print('tags in S3', bucket_contents)

  local_tags = check_output(['git', 'tag', '-l', f'{constants.BUILD_TAG_PREFIX}*'])
  local_tags = local_tags.decode("utf-8").strip().split('\n')
  local_tags = map(lambda x: x.strip(constants.BUILD_TAG_PREFIX), local_tags)
  local_tags = filter(lambda x: len(x) > 0, local_tags)
  local_tags = set(local_tags)
  print('local_tags', local_tags)

  intersection = local_tags - bucket_contents
  print('intersection:', intersection)

  return [*intersection]


def build_tag(tag):
  call(['git', '-c', 'advice.detachedHead=false', 'checkout', f'{constants.BUILD_TAG_PREFIX}{tag}'])
  # fetch dataset at this commit if not cached already
  verify_and_download()
  for file in constants.BUILD_PRODUCTS:
    try_to_remove(file)
  out = check_output(['bash', 'scripts/run_container.sh', constants.PYTHON_VERSION, constants.NOTEBOOK_NAME, constants.DOCKER_IMAGE_NAME, *constants.VOLUME_MAPPINGS])
  print(out)

def push_tag (tag):
  for file in constants.BUILD_PRODUCTS:
    extra_args = {
        'ACL': 'public-read',
      }
    if file == 'index.html':
      extra_args = {
        'ACL': 'public-read',
        'ContentType': 'text/html',
        'ContentDisposition': 'inline'
      }

    s3_client.upload_file(
      f'{constants.WORKING_DIR}/{file}',
      constants.S3_BUCKET,
      f'{constants.S3_BUCKET_DIR}/{tag}/{file}',
      ExtraArgs=extra_args
    )
    
    try_to_remove(file)


# ------------------------------------------------

if __name__ == "__main__":
  dt_string = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
  print('<<<<<<<<< START OF EXECUTION >>>>>>>>>  ', dt_string)	

  for tag in tags_to_build():
    try:
      print(f'Now checking out and building tag labled {tag}')
      
      build_tag(tag)

      push_tag(tag)

      print('----------------------')
    except Exception as e:      
      with open('index.html', 'w') as file:
        file.write(str(e))
        traceback.print_tb(sys.exc_info()[-1], limit=None, file=file)
      s3_client.upload_file(
        'index.html',
        constants.S3_BUCKET,
        f'{constants.S3_BUCKET_DIR}/{tag}/index.html',
        ExtraArgs={
          'ACL': 'public-read',
          'ContentType': 'text/html',
          'ContentDisposition': 'inline'
        }
      )
  print('Returning to master branch')
  call(['git', 'checkout', 'master'])
  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END OF EXECUTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n')
