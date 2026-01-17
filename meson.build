project('libsmarter')

incl = include_directories('include/')

install_headers('include/smarter.hpp')

libsmarter_dep = declare_dependency(include_directories: incl)

pkgconfig = import('pkgconfig')
pkgconfig.generate(
  name: 'libsmarter',
  description: 'A library for freestanding smart pointers in C++',
  version: meson.project_version(),
)
