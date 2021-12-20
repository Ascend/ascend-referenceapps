/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCEND_TENSOR_INCLUDED
#define ASCEND_TENSOR_INCLUDED

#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <assert.h>
#include <initializer_list>

namespace ascend {
const int DIMS_1 = 1;
const int DIMS_2 = 2;
const int DIMS_3 = 3;
const int DIMS_4 = 4;
const int DIMS_5 = 5;

namespace traits {
template<typename T>
struct RestrictPtrTraits {
    typedef T *__restrict__ PtrType;
};

template<typename T>
struct DefaultPtrTraits {
    typedef T *PtrType;
};
} // namespace traits

template<typename T,
          int Dim,
          typename IndexT,
          template<typename U> class PtrTraits>
class AscendTensor;

namespace detail {
template<typename TensorType, int Subdim, template<typename U> class PtrTraits>
class SubTensor;
}

template<typename T,
          int Dim,
          typename IndexT = int,
          template<typename U> class PtrTraits = traits::DefaultPtrTraits>
class AscendTensor {
public:
    typedef T DataType;
    typedef IndexT IndexType;
    typedef typename PtrTraits<T>::PtrType DataPtrType;
    typedef AscendTensor<T, Dim, IndexT, PtrTraits> TensorType;

    enum {
        NUM_DIM = Dim
    };

    // Default constructor
    AscendTensor();

    // Destructor
    ~AscendTensor();

    // Copy constructor
    AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &t);

    // Move constructor
    AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &&t);

    // Copy assignment
    AscendTensor<T, Dim, IndexT, PtrTraits> &operator = (const AscendTensor<T, Dim, IndexT, PtrTraits> &t);

    // Move assignment
    AscendTensor<T, Dim, IndexT, PtrTraits> &operator = (AscendTensor<T, Dim, IndexT, PtrTraits> &&t);

    // Constructs a tensor of the given size and stride, referencing a
    // memory region we do not own
    AscendTensor(DataPtrType data, const IndexT sizes[Dim]);
    AscendTensor(DataPtrType data, std::initializer_list<IndexT> sizes);

    // Constructs a tensor of the given size and stride, referencing a
    // memory region we do not own
    AscendTensor(DataPtrType data, const IndexT sizes[Dim], const IndexT strides[Dim]);

    // Constructs a tensor of the given size, allocating memory for it locally
    AscendTensor(const IndexT sizes[Dim]);
    AscendTensor(std::initializer_list<IndexT> sizes);

    // Constructs a tensor of the given size, reserving a temporary
    // memory reservation via a memory manager.
    // The memory reservation should be ordered with respect to the
    // given stream.
    AscendTensor(AscendMemory &m, const IndexT sizes[Dim], aclrtStream stream);
    AscendTensor(AscendMemory &m, std::initializer_list<IndexT> sizes, aclrtStream stream);

    // Copies a tensor into ourselves; sizes must match
    void copyFrom(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream);

    // Copies ourselves into a tensor; sizes must match
    void copyTo(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream);

    // Copies a tensor into ourselves; sizes must match
    void copyFromSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t);

    // Copies ourselves into a tensor; sizes must match
    void copyToSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t);

    // Call to zero out memory
    AscendTensor<T, Dim, IndexT, PtrTraits> &zero();

    // Cast to a tensor of a different type of the same size and
    // stride. U and our type T must be of the same size
    template<typename U>
    AscendTensor<U, Dim, IndexT, PtrTraits> cast();

    // Const cast to a tensor of a different type of the same size and
    // stride. U and our type T must be of the same size
    template<typename U>
    AscendTensor<U, Dim, IndexT, PtrTraits> cast() const;

    inline int dimNum() const;

    inline const IndexT *sizes() const;

    inline const IndexT *strides() const;

    DataPtrType data() const;

    DataPtrType end() const;

    IndexT getSize(int i) const;

    IndexT getStride(int i) const;

    size_t numElements() const;

    size_t getSizeInBytes() const;

    void initValue(DataType val);

    // Returns a tensor that is a view of the `SubDim`-dimensional slice
    // of this tensor, starting at `at`.
    template<int SubDim>
    AscendTensor<T, SubDim, IndexT, PtrTraits> view(DataPtrType data);

    // Returns a tensor that is a view of the `SubDim`-dimensional slice
    // of this tensor, starting where our data begins
    template<int SubDim>
    AscendTensor<T, SubDim, IndexT, PtrTraits> view();

    // Returns a view of the given tensor expressed as a tensor of a
    // different number of dimensions.
    // Only works if we are contiguous.
    template<int NewDim>
    AscendTensor<T, NewDim, IndexT, PtrTraits> view(std::initializer_list<IndexT> sizes);

    // Returns a read/write view of a portion of our tensor.
    detail::SubTensor<TensorType, Dim - 1, PtrTraits> operator[](IndexT index);

    // Returns a const view of a portion of our tensor.
    const detail::SubTensor<TensorType, Dim - 1, PtrTraits> operator[](IndexT index) const;

private:
    DataPtrType dataPtr;

    IndexT strideArray[Dim];

    IndexT sizeArray[Dim];

    enum AllocState {
        ALLOC_STATE_OWNER,
        ALLOC_STATE_NOT_OWNER,
        ALLOC_STATE_RESERVATION
    };

    AllocState state;
    AscendMemoryReservation reservation;
};

namespace detail {
template<typename TensorType, template<typename U> class PtrTraits>
class SubTensor<TensorType, 0, PtrTraits> {
public:
    SubTensor<TensorType, 0, PtrTraits> operator = (typename TensorType::DataType val);

    operator typename TensorType::DataType &();

    operator const typename TensorType::DataType &() const;

    typename TensorType::DataType *operator&();

    const typename TensorType::DataType *operator&() const;

    // Returns a raw accessor to our slice.
    typename TensorType::DataPtrType data();

    // Returns a raw accessor to our slice (const).
    const typename TensorType::DataPtrType data() const;

    typename TensorType::DataType value();

    const typename TensorType::DataType value() const;

    // Cast to a different datatype.
    template<typename T>
    T &as();

    // Cast to a different datatype(const).
    template<typename T>
    const T &as() const;

    // Cast to a different datatype
    template<typename T>
    typename PtrTraits<T>::PtrType dataAs();

    // Cast to a different datatype(const)
    template<typename T>
    typename PtrTraits<const T>::PtrType dataAs() const;

protected:
    friend class SubTensor<TensorType, 1, PtrTraits>;

    friend class AscendTensor<typename TensorType::DataType,
                              1,
                              typename TensorType::IndexType,
                              PtrTraits>;

    SubTensor(TensorType &t, typename TensorType::DataPtrType d);

    TensorType &tensor;

    typename TensorType::DataPtrType const dataPtr;
};

// A `SubDim`-rank slice of a parent Tensor
template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
class SubTensor {
public:
    // Returns a view of the data located at our offset (the dimension
    // `SubDim` - 1 tensor).
    inline SubTensor<TensorType, SubDim - 1, PtrTraits> operator[](typename TensorType::IndexType index);

    // Returns a view of the data located at our offset (the dimension
    // `SubDim` - 1 tensor)(const).
    inline const SubTensor<TensorType, SubDim - 1, PtrTraits> operator[](typename TensorType::IndexType index) const;

    // operator& returning T*
    typename TensorType::DataType *operator&();

    // const operator& returning const T*
    const typename TensorType::DataType *operator&() const;

    // Returns a raw accessor to our slice.
    typename TensorType::DataPtrType data();

    // Returns a raw accessor to our slice (const).
    const typename TensorType::DataPtrType data() const;

    // Cast to a different datatype.
    template<typename T>
    T &as();

    // Cast to a different datatype(const).
    template<typename T>
    const T &as() const;

    // Cast to a different datatype
    template<typename T>
    typename PtrTraits<T>::PtrType dataAs();

    // Cast to a different datatype(const)
    template<typename T>
    typename PtrTraits<const T>::PtrType dataAs() const;

    AscendTensor<typename TensorType::DataType, SubDim, typename TensorType::IndexType, PtrTraits>
    view();

protected:
    // One dimension greater can create us
    friend class SubTensor<TensorType, SubDim + 1, PtrTraits>;

    /// Our parent tensor can create us
    friend class AscendTensor<typename TensorType::DataType,
                              TensorType::NUM_DIM,
                              typename TensorType::IndexType,
                              PtrTraits>;

    SubTensor(TensorType &t, typename TensorType::DataPtrType data);

    // The tensor we're referencing
    TensorType &tensor;

    // Where our value is located
    typename TensorType::DataPtrType const dataPtr;
};
} // namespace detail
} // namespace ascend
#include <ascenddaemon/utils/AscendTensorInl.h>

#endif // ASCEND_TENSOR_INCLUDED
