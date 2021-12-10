%module ascendfaiss;

%include <stdint.i>
typedef int64_t size_t;

#define __restrict // necessary

%{

#include <stdint.h>

#ifdef SWIGPYTHON

#undef popcount64

#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#endif

#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendIndexIVFPQ.h>
#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <faiss/ascend/AscendIndexIVFFlat.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/AscendIndexInt8IVFFlat.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendVectorTransform.h>
#include <faiss/ascend/AscendIndexPreTransform.h>
#include <faiss/ascend/AscendAutoTune.h>
#include <faiss/impl/AuxIndexStructures.h>

%}

/********************************************************
 * GIL manipulation and exception handling
 ********************************************************/

#ifdef SWIGPYTHON
// %catches(faiss::FaissException);


// Python-specific: release GIL by default for all functions
%exception {
    Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(faiss::FaissException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc & ba) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    }
    Py_END_ALLOW_THREADS
}

#endif

/*******************************************************************
 * Types of vectors we want to manipulate at the scripting language
 * level.
 *******************************************************************/

// simplified interface for vector
namespace std {

    template<class T>
    class vector {
    public:
        vector();
        void push_back(T);
        void clear();
        T * data();
        size_t size();
        T at (size_t n) const;
        void resize (size_t n);
        void swap (vector<T> & other);
    };
};

%include <std_string.i>

%template(FloatVector) std::vector<float>;
%template(DoubleVector) std::vector<double>;
%template(ByteVector) std::vector<uint8_t>;
%template(CharVector) std::vector<char>;
// NOTE(hoss): Using unsigned long instead of uint64_t because OSX defines
//   uint64_t as unsigned long long, which SWIG is not aware of.
%template(Uint64Vector) std::vector<unsigned long>;
%template(LongVector) std::vector<long>;
%template(IntVector) std::vector<int>;
%template(FloatVectorVector) std::vector<std::vector<float> >;
%template(ByteVectorVector) std::vector<std::vector<unsigned char> >;
%template(CharVectorVector) std::vector<std::vector<char> >;
%template(LongVectorVector) std::vector<std::vector<long> >;

%ignore faiss::ascend::AscendIndexConfig::AscendIndexConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFConfig::AscendIndexIVFConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFPQConfig::AscendIndexIVFPQConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFSQConfig::AscendIndexIVFSQConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFFlatConfig::AscendIndexIVFFlatConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexInt8Config::AscendIndexInt8Config(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexInt8IVFConfig::AscendIndexInt8IVFConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexInt8IVFFlatConfig::AscendIndexInt8IVFFlatConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexInt8FlatConfig::AscendIndexInt8FlatConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexSQConfig::AscendIndexSQConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexFlatConfig::AscendIndexFlatConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::ToAscendCloner;
%ignore faiss::ascend::ToCPUCloner;

%include <faiss/Index.h>
%include <faiss/AutoTune.h>
%include <faiss/Clustering.h>
%include <faiss/ascend/AscendIndex.h>
%include <faiss/ascend/AscendIndexIVF.h>
%include <faiss/ascend/AscendIndexIVFPQ.h>
%include <faiss/ascend/AscendIndexIVFSQ.h>
%include <faiss/ascend/AscendIndexIVFFlat.h>
%include <faiss/ascend/AscendIndexInt8.h>
%include <faiss/ascend/AscendIndexInt8Flat.h>
%include <faiss/ascend/AscendIndexInt8IVF.h>
%include <faiss/ascend/AscendIndexInt8IVFFlat.h>
%include <faiss/ascend/AscendIndexFlat.h>
%include <faiss/ascend/AscendIndexSQ.h>
%include <faiss/clone_index.h>
%include <faiss/ascend/AscendClonerOptions.h>
%include <faiss/ascend/AscendAutoTune.h>

%include <faiss/ascend/AscendIndexPreTransform.h>
%extend faiss::ascend::AscendIndexPreTransform {
    template<class Transform>
    void prepend(int dimIn, int dimOut, float eigenPower, bool randomRotation) {
        $self->prependTransform<Transform>(dimIn, dimOut, eigenPower, randomRotation);
    }

    %template(prependPCA) prepend<faiss::ascend::AscendPCAMatrix>;
}

%ignore faiss::InvertedListScanner;
%ignore faiss::ScalarQuantizer::SQDistanceComputer;
%ignore faiss::ScalarQuantizer::Quantizer;
%ignore faiss::ScalarQuantizer::select_InvertedListScanner;
%ignore faiss::ScalarQuantizer::get_distance_computer;
%include <faiss/impl/ScalarQuantizer.h>

%ignore faiss::BufferList::Buffer;
%ignore faiss::RangeSearchPartialResult::QueryResult;
%ignore faiss::IDSelectorBatch::set;
%ignore faiss::IDSelectorBatch::bloom;
%ignore faiss::InterruptCallback::instance;
%ignore faiss::InterruptCallback::lock;
%include <faiss/impl/AuxIndexStructures.h>


#ifdef SWIGPYTHON

%define DOWNCAST(subclass)
    if (dynamic_cast<faiss::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ ## subclass,$owner);
    } else
%enddef

%define DOWNCAST2(subclass, longname)
    if (dynamic_cast<faiss::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ ## longname,$owner);
    } else
%enddef

%define DOWNCAST_ASCEND(subclass)
    if (dynamic_cast<faiss::ascend::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ascend__ ## subclass,$owner);
    } else
%enddef

#endif

// Subclasses should appear before their parent
%typemap(out) faiss::Index * {
    DOWNCAST_ASCEND ( AscendIndexFlat )
    DOWNCAST_ASCEND ( AscendIndexIVFPQ )
    DOWNCAST_ASCEND ( AscendIndexIVFSQ )
    DOWNCAST_ASCEND ( AscendIndexIVFFlat )
    DOWNCAST_ASCEND ( AscendIndexPreTransform )

    // default for non-recognized classes
    DOWNCAST ( Index )
    if ($1 == NULL)
    {
#ifdef SWIGPYTHON
        $result = SWIG_Py_Void();
#endif
    } else {
        assert(false);
    }
}

%typemap(out) faiss::ascend::AscendIndexInt8 * {
    DOWNCAST_ASCEND ( AscendIndexInt8Flat )
    DOWNCAST_ASCEND ( AscendIndexInt8IVFFlat )
    if ($1 == NULL)
    {
#ifdef SWIGPYTHON
        $result = SWIG_Py_Void();
#endif
    } else {
        assert(false);
    }
}

%inline %{
faiss::Index * downcast_index (faiss::Index *index)
{
    return index;
}
%}

%newobject index_ascend_to_cpu;
%newobject index_cpu_to_ascend;
%newobject index_int8_ascend_to_cpu;
%newobject index_int8_cpu_to_ascend;
%include <faiss/ascend/AscendCloner.h>

// Python-specific: do not release GIL any more, as functions below
// use the Python/C API
#ifdef SWIGPYTHON
%exception;
#endif


/*******************************************************************
 * Python specific: numpy array <-> C++ pointer interface
 *******************************************************************/

#ifdef SWIGPYTHON

%{
PyObject *swig_ptr (PyObject *a)
{
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long_long, 0);
#endif
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long_long, 0);
#endif
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}

%}


%init %{
    /* needed, else crash at runtime */
    import_array();

%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(ctype, numpytype)

%{
PyObject * rev_swig_ptr(ctype *src, npy_intp size) {
    return PyArray_SimpleNewFromData(1, &size, numpytype, src);
}
%}

PyObject * rev_swig_ptr(ctype *src, size_t size);

%enddef

REV_SWIG_PTR(float, NPY_FLOAT32);
REV_SWIG_PTR(int, NPY_INT32);
REV_SWIG_PTR(unsigned char, NPY_UINT8);
REV_SWIG_PTR(char, NPY_INT8);
REV_SWIG_PTR(int64_t, NPY_INT64);
REV_SWIG_PTR(uint64_t, NPY_UINT64);

#endif
