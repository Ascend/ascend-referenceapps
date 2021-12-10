# -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([ascendfaiss], [1.0])
AC_COPYRIGHT([Copyright (C) <2019>  <Huawei Technologies Co., Ltd.>. All Rights Reserved.])
AC_CONFIG_SRCDIR([ascend/AscendIndex.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([acinclude])

AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE

: ${CXXFLAGS="-g -O3 -Wall -Wextra"}

# Checks for programs.
AC_LANG(C++)
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory])
AC_PROG_CPP
AC_PROG_MAKE_SET
AC_PROG_MKDIR_P
AC_PROG_INSTALL

FA_PYTHON

if test x$PYTHON != x; then
   FA_NUMPY
fi

FA_PROG_SWIG
FA_CHECK_FAISS
FA_CHECK_ASCEND

# Checks for header files.
AC_CHECK_HEADERS([float.h limits.h stddef.h stdint.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_C_RESTRICT
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_CHECK_FUNCS([clock_gettime floor gettimeofday memmove memset munmap pow sqrt strerror strstr])

AC_OPENMP

AX_CPU_ARCH

AC_CONFIG_FILES([makefile.inc])
AC_OUTPUT
