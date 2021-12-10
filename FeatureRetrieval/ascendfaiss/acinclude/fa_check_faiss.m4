AC_DEFUN([FA_CHECK_FAISS], [

AC_ARG_WITH(faiss,
  [AS_HELP_STRING([--with-faiss=<faiss-path>], [installation path of faiss])],
  [],
  [with_faiss=/usr/local])

if test x$with_faiss != x; then
    faiss_prefix=$with_faiss
    FAISS_LIBS=-lfaiss
    FAISS_CPPFLAGS=-I$with_faiss/include
    FAISS_LDFLAGS=-L$with_faiss/lib
else
    faiss_prefix=/usr/local
    FAISS_LIBS=-lfaiss
    FAISS_CPPFLAGS=-I$faiss_prefix/include/
    FAISS_LDFLAGS=-L$faiss_prefix/lib/
fi

AC_CHECK_HEADER([faiss/Index.h], [], AC_MSG_FAILURE([Please check faiss installation path to find Index.h]))

AC_SUBST(FAISS_CPPFLAGS)
AC_SUBST(FAISS_LDFLAGS)
AC_SUBST(FAISS_LIBS)
AC_SUBST(FAISS_PREFIX, $faiss_prefix)
])
