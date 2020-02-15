BUILD_TAG_PREFIX = 'build_'
S3_BUCKET = 'built-model-repository'
S3_BUCKET_DIR = 'intro_to_ai_python'
DATASET_URL = 'https://s3-us-west-2.amazonaws.com/datasets-349058029/IntroToAIPython/flowers.zip'
# refactor to dataset url: dir mapping and iterate in get dataset script?
DATASET_DESTINATION = 'src/flowers'
PYTHON_VERSION = 'python3.6'
DOCKER_IMAGE_NAME = 'intro_ai'
VOLUME_MAPPINGS = {
  '/src': '/src',
  '/container_cache/torch': '/root/.cache/torch/checkpoints'
}
BUILLD_PRODUCTS = []
# use to customize what products to clear out 
