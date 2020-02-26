import constants
from subprocess import call

print(['running teardown script'])
call(['docker', 'rmi', '--force', constants.DOCKER_IMAGE_NAME])