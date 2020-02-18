WORKING_DIR = 'src/datasets'
BUILD_TAG_PREFIX = 'build_'
S3_BUCKET = 'built-model-repository'
S3_BUCKET_DIR = 'intro_to_ai_python'
PYTHON_VERSION = 'python3.6'
DOCKER_IMAGE_NAME = 'intro_ai'
VOLUME_MAPPINGS = {
  '/src': '/src',
  '/container_cache/torch': '/root/.cache/torch/checkpoints'
}
BUILLD_PRODUCTS = ['checkpoint.pt', 'notebook.html']

# I'd like to figure out a way to move this file to the root of the project raterh than in the scripts file