import boto3
import constants
from  os import path, remove
from subprocess import call, check_output
from get_dataset import verify_and_download

# This script assupes you have the package 'boto3' installed and accessible from here

def filter_by_dir_name(s3_object):
  if s3_object.key.find(constants.S3_BUCKET_DIR) > -1:
    path = s3_object.key.split('/')
    return (len(path) > 1 and len(path[1]) > 0)
  else:
    return False


def extract_child_dirs(s3_object):
  path = s3_object.key.split('/')
  return path[1]


def tags_to_build ():
  s3 = boto3.resource('s3')
  my_bucket = s3.Bucket(constants.S3_BUCKET)
  
  bucket_contents = [*my_bucket.objects.all()]
  bucket_contents = filter(filter_by_dir_name, bucket_contents)
  bucket_contents = map(extract_child_dirs, bucket_contents)
  bucket_contents = set(bucket_contents)
  print('tags in S3', bucket_contents)

  local_tags = check_output(['git', 'tag', '-l', 'build_*'])
  local_tags = local_tags.decode("utf-8").strip().split('\n')
  local_tags = map(lambda x: x.strip('build_'), local_tags)
  local_tags = set(local_tags)
  print('local_tags', local_tags)

  intersection = local_tags - bucket_contents
  print('intersection:', intersection)

  return [*intersection]


def build_tag(tag):
  # call(['git', 'checkout', f'build_{tag}'])
  # Remove any previous build products, if they exist
  for file in constants.BUILLD_PRODUCTS:
    try:
      remove(f'{constants.WORKING_DIR}/{file}')
    except FileNotFoundError:
      pass
    
  call(['bash', 'scripts/run_container.sh', constants.PYTHON_VERSION, f'Hell000 {tag}'])
  # your current tags point to commits where these files don't exist
  # update the tag locations or comment out the checking out step for now


def push_tag (tag):
  # PUSH to S3 WITH CORRECT PERMISSIONS
  # Remove any previous build products
  for file in constants.BUILLD_PRODUCTS:
    try:
      remove(f'{constants.WORKING_DIR}/{file}')
    except FileNotFoundError:
      pass


# ------------------------------------------------

if __name__ == "__main__":

  # # run tag analyzer (accept s3 url)
  tags = tags_to_build()

  for tag in tags:
    print(tag)
    # fetch dataset at this commit if not cached already
    verify_and_download()
    build_tag(tag)
    # invoke push script (accept tag name)

  call(['git', 'checkout', 'master'])
