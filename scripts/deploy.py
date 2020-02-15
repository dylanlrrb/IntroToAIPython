from subprocess import call

def build_tag(tag):
  call(['git', 'checkout', f'build_{tag}'])

def push_tag (tag):
  pass