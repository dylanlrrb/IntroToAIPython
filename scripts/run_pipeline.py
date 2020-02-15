from subprocess import call
from get_dataset import verify_and_download
from analyze_tags import tags_to_build
from build import build_tag

# # fetch dataset (accept dataset url and dataset destination)
verify_and_download()

# # run tag analyzer (accept s3 url)
tags = tags_to_build()

for tag in tags:
  print(tag)
  build_tag(tag)
  # invole push script (accept tag name)

call(['git', 'checkout', 'master'])

# Prefix a tag with "build_" then the version number
# suffix with "_<rebuild number>" to trigger a rebuild on a particular commit
# "git push origin master --tags" to push the tags up to github to be pulled down later
# git tag -l "build*" to list all tags that match the pattern