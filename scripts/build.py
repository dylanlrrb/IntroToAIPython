from subprocess import call

def build_tag(tag):
  call(['git', 'checkout', tag])