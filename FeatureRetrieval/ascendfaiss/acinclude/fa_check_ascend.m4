AC_DEFUN([FA_CHECK_ASCEND], [

AC_ARG_WITH(ascend,
  [AS_HELP_STRING([--with-ascend=<ascend-acllib-path>], [installation path of ascend acllib])],
  [],
  [with_ascend=/usr/local/Ascend/ascend-toolkit/latest])

AC_ARG_WITH(ascendminios,
  [AS_HELP_STRING([--with-ascendminios=<ascend-minios-path>], [installation path of ascend minios])],
  [],
  [with_ascendminios=/usr/local/AscendMiniOs])

AC_ARG_WITH(ascenddriver,
  [AS_HELP_STRING([--with-ascenddriver=<ascend-driver-path>], [installation path of ascend driver])],
  [],
  [with_ascenddriver=/usr/local/Ascend])

AC_ARG_WITH(protobuf,
  [AS_HELP_STRING([--with-protobuf=<protobuf-path>], [installation path of protobuf])],
  [],
  [with_protobuf=/usr/local/protobuf])

AC_ARG_WITH(protobufaarch64,
  [AS_HELP_STRING([--with-protobufaarch64=<protobuf-aarch64-path>], [installation path of protobuf aarch64])],
  [],
  [with_protobufaarch64=/opt/aarch64/protobuf])

AC_CANONICAL_TARGET
case $target in
    amd64-* | x86_64-*)
      arch_path=hcc
      tool_prefix=aarch64-target-linux-gnu
      ;;
    aarch64*-*)
      arch_path=hcc
      tool_prefix=aarch64-target-linux-gnu
      ;;
    *) ;;
  esac

if test x$with_ascendminios != x; then
    ascendminios_prefix=$with_ascendminios
fi

if test x$with_ascenddriver != x; then
    ascenddriver_prefix=$with_ascenddriver
fi

if test x$with_protobuf != x; then
    protobuf_prefix=$with_protobuf
fi

if test x$with_protobufaarch64 != x; then
    protobuf_aarch64_prefix=$with_protobufaarch64
fi

AC_CHECK_FILE([$protobuf_prefix/lib/libprotobuf.so], [], AC_MSG_FAILURE([Please check installation path of protobuf for host]))

AC_CHECK_FILE([$protobuf_aarch64_prefix/lib/libprotobuf.a], [], AC_MSG_FAILURE([Please check installation path of protobuf aarch64 for device]))

if test x$with_ascend != x; then
    ascend_prefix=$with_ascend
    AC_CHECK_PROG(ASCEND_CXX, [$tool_prefix-g++], [$ascend_prefix/toolkit/toolchain/$arch_path/bin/$tool_prefix-g++], 
    	[], [$ascend_prefix/toolkit/toolchain/$arch_path/bin/])
    AC_CHECK_PROG(ASCEND_AR, [$tool_prefix-ar], [$ascend_prefix/toolkit/toolchain/$arch_path/bin/$tool_prefix-ar], 
        [], [$ascend_prefix/toolkit/toolchain/$arch_path/bin/])
    ASCEND_FLAGS="-DUSE_ACL_INTERFACE_V2 -I$ascendminios_prefix/acllib/include/ -I$ascenddriver_prefix/driver/include/dvpp/ -I$ascenddriver_prefix/driver/kernel/inc/driver -I$ascenddriver_prefix/driver/kernel/libc_sec/include"
    ASCEND_LDFLAGS="-L$ascendminios_prefix/acllib/lib64/stub -L$ascenddriver_prefix/develop/lib64 -L$ascenddriver_prefix/driver/lib64 -Wl,--rpath-link=$ascenddriver_prefix/develop/lib64:$ascenddriver_prefix/driver/lib64:$ascendminios_prefix/acllib/lib64"
    ASCEND_LIBS="-lascendcl -lascend_hal -lc_sec $protobuf_aarch64_prefix/lib/libprotobuf.a"
    ASCEND_HOSTFLAGS="-I$ascenddriver_prefix/driver/kernel/inc/driver -I$ascenddriver_prefix/driver/kernel/libc_sec/include"
    ASCEND_HOSTLDFLAGS="-L$ascenddriver_prefix/driver/lib64 -L$protobuf_prefix/lib -Wl,-rpath-link=$ascenddriver_prefix/driver/lib64:$protobuf_prefix/lib"
    ASCEND_HOSTLIBS="-lascend_hal -lc_sec -lprotobuf"
else
    AC_CHECK_PROGS(ASCEND_CXX, [$tool_prefix-g++ $ascend_prefix/toolkit/toolchain/$arch_path/bin/$tool_prefix-g++], [])
    AC_CHECK_PROGS(ASCEND_AR, [$tool_prefix-ar], [$ascend_prefix/toolkit/toolchain/$arch_path/bin/$tool_prefix-ar], [], [$ascend_prefix/toolkit/toolchain/$arch_path/bin/])
    if test "x$ASCEND_CXX" == "x/usr/local/Ascend/toolkit/toolchain/$arch_path/bin/$tool_prefix-g++"; then
      ascend_prefix="/usr/local/Ascend"
      ASCEND_FLAGS="-DUSE_ACL_INTERFACE_V2 -I$ascendminios_prefix/acllib/include/ -I$ascenddriver_prefix/driver/include/dvpp/ -I$ascenddriver_prefix/driver/kernel/inc/driver -I$ascenddriver_prefix/driver/kernel/libc_sec/include"
      ASCEND_LDFLAGS="-L$ascendminios_prefix/acllib/lib64/stub -L$ascenddriver_prefix/develop/lib64 -L$ascenddriver_prefix/driver/lib64 -Wl,--rpath-link=$ascenddriver_prefix/develop/lib64:$ascenddriver_prefix/driver/lib64:$ascendminios_prefix/acllib/lib64"
      ASCEND_LIBS="-lascendcl -lascend_hal -lc_sec $protobuf_aarch64_prefix/lib/libprotobuf.a"
      ASCEND_HOSTFLAGS="-I$ascenddriver_prefix/driver/kernel/inc/driver -I$ascenddriver_prefix/driver/kernel/libc_sec/include"
      ASCEND_HOSTLDFLAGS="-L$ascenddriver_prefix/driver/lib64 -L$protobuf_prefix/lib -Wl,-rpath-link=$ascenddriver_prefix/driver/lib64:$protobuf_prefix/lib"
      ASCEND_HOSTLIBS="-lascend_hal -lc_sec -lprotobuf"
    else
      ascend_prefix=""
      ASCEND_FLAGS=""
      ASCEND_LDFLAGS=""
      ASCEND_LIBS=""
      ASCEND_HOSTFLAGS=""
      ASCEND_HOSTLDFLAGS=""
      ASCEND_HOSTLIBS=""
    fi
fi

if test "x$ASCEND_CXX" == x; then
	AC_MSG_ERROR([Couldn't find $tool_prefix-g++, please check installation path of toolkit in CANN])
fi

if test "x$ASCEND_AR" == x; then
	AC_MSG_ERROR([Couldn't find $tool_prefix-ar, please check installation path of toolkit in CANN])
fi

fa_save_CXX="$CXX"
fa_save_CPPFLAGS="$CPPFLAGS"
fa_save_LDFLAGS="$LDFLAGS"
fa_save_LIBS="$LIBS"

CXX="$ASCEND_CXX"
CPPFLAGS="$ASCEND_FLAGS $CPPFLAGS"
LDFLAGS="$ASCEND_LDFLAGS $LDFLAGS"
LIBS="$ASCEND_LIBS $LIBS"

AC_CHECK_HEADER([acl/acl.h], [], AC_MSG_FAILURE([Please check acllib minios.aarch64 installation path to find acl.h]))
#AC_CHECK_LIB([ascendcl], [aclrtMalloc], [], AC_MSG_FAILURE([Couldn't find libascendcl.so]))

ASCEND_FLAGS="$CPPFLAGS"
ASCEND_LDFLAGS="$LDFLAGS"
ASCEND_LIBS="$LIBS"

CXX="$fa_save_CXX"
CPPFLAGS="$fa_save_CPPFLAGS"
LDFLAGS="$fa_save_LDFLAGS"
LIBS="$fa_save_LIBS"



fa_save_CPPFLAGS="$CPPFLAGS"
fa_save_LDFLAGS="$LDFLAGS"
fa_save_LIBS="$LIBS"

CPPFLAGS="$ASCEND_HOSTFLAGS $CPPFLAGS"
LDFLAGS="$ASCEND_HOSTLDFLAGS $LDFLAGS"
LIBS="$ASCEND_HOSTLIBS $LIBS"

AC_CHECK_HEADER([ascend_hal.h], [], AC_MSG_FAILURE([Please check Ascend310-driver installation path to find ascend_hal.h]))
#AC_CHECK_LIB([ascend_hal], [drvHdcAllocMsg], [], AC_MSG_FAILURE([Couldn't find libascend_hal.so]))

ASCEND_HOSTFLAGS="$CPPFLAGS"
ASCEND_HOSTLDFLAGS="$LDFLAGS"
ASCEND_HOSTLIBS="$LIBS"

CPPFLAGS="$fa_save_CPPFLAGS"
LDFLAGS="$fa_save_LDFLAGS"
LIBS="$fa_save_LIBS"

AC_SUBST(ASCEND_HOSTFLAGS)
AC_SUBST(ASCEND_HOSTLDFLAGS)
AC_SUBST(ASCEND_HOSTLIBS)
AC_SUBST(ASCEND_CXX)
AC_SUBST(ASCEND_AR)
AC_SUBST(ASCEND_FLAGS)
AC_SUBST(ASCEND_LDFLAGS)
AC_SUBST(ASCEND_LIBS)
AC_SUBST(ASCEND_PREFIX, $ascend_prefix)
AC_SUBST(DRIVER_PREFIX, $ascenddriver_prefix)
AC_SUBST(PROTOBUF_PREFIX, $protobuf_prefix)
AC_SUBST(PROTOBUF_AARCH64_PREFIX, $protobuf_aarch64_prefix)
])
