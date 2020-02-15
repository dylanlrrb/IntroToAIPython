from subprocess import call
from get_dataset import verify_and_download
from analyze_tags import tags_to_build

# # fetch dataset (accept dataset url and dataset destination)
verify_and_download()

# # run tag analyzer (accept s3 url)
tags = tags_to_build()

# # for tag in tags:
# #   invoke the build scripts (accept tag name)
# #   invole push script (accept tag name, s3 url)

