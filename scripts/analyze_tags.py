import boto3
import constants

from subprocess import check_output


s3 = boto3.resource('s3')
my_bucket = s3.Bucket(constants.S3_BUCKET)

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
  bucket_contents = [*my_bucket.objects.all()]
  bucket_contents = filter(filter_by_dir_name, bucket_contents)
  bucket_contents = map(extract_child_dirs, bucket_contents)
  bucket_contents = set(bucket_contents)
  print('tags in S3', bucket_contents)

  local_tags = check_output(['git', 'tag', '-l', 'build*'])
  local_tags = local_tags.decode("utf-8").strip().split('\n')
  local_tags = map(lambda x: x.strip('build_'), local_tags)
  local_tags = set(local_tags)
  print('local_tags', local_tags)

  intersection = local_tags - bucket_contents
  print('intersection:', intersection)

  return [*intersection]