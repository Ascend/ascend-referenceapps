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

#ifndef ASCEND_TENSOR_INL_INCLUDED
#define ASCEND_TENSOR_INL_INCLUDED

namespace ascend {
template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor() : dataPtr(nullptr), state(ALLOC_STATE_NOT_OWNER)
{
    static_assert(Dim > 0, "must have > 0 dimensions");
    for (int i = 0; i < Dim; ++i) {
        sizeArray[i] = 0;
        strideArray[i] = (IndexT)1;
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::~AscendTensor()
{
    if (state == ALLOC_STATE_OWNER && this->dataPtr != nullptr) {
        FreeMemorySpace(MemorySpace::DEVICE, (void *)this->dataPtr);
        this->dataPtr = nullptr;
    }

    // Otherwise, if we have a temporary memory reservation, then its
    // destructor will return the reservation
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &t)
{
    this->operator = (t);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &&t)
{
    this->operator = (std::move(t));
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(DataPtrType data, const IndexT sizes[Dim])
    : dataPtr(data), state(ALLOC_STATE_NOT_OWNER)
{
    static_assert(Dim > 0, "must have > 0 dimensions");

    for (int i = 0; i < Dim; ++i) {
        sizeArray[i] = sizes[i];
    }

    strideArray[Dim - 1] = static_cast<IndexT>(1);
    const int offset = 2;
    for (int i = Dim - offset; i >= 0; --i) {
        strideArray[i] = strideArray[i + 1] * sizeArray[i + 1];
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(DataPtrType data, std::initializer_list<IndexT> sizes)
    : dataPtr(data), state(ALLOC_STATE_NOT_OWNER)
{
    static_assert(Dim > 0, "must have > 0 dimensions");

    int i = 0;
    for (auto s : sizes) {
        sizeArray[i] = s;
        i++;
    }

    strideArray[Dim - 1] = static_cast<IndexT>(1);
    const int offset = 2;
    for (i = Dim - offset; i >= 0; --i) {
        strideArray[i] = strideArray[i + 1] * sizeArray[i + 1];
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(DataPtrType data, const IndexT sizes[Dim],
    const IndexT strides[Dim])
    : dataPtr(data), state(ALLOC_STATE_NOT_OWNER)
{
    for (int i = 0; i < Dim; ++i) {
        sizeArray[i] = sizes[i];
        strideArray[i] = strides[i];
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(const IndexT sizes[Dim]) : AscendTensor(nullptr, sizes)
{
    this->state = ALLOC_STATE_OWNER;
    allocMemorySpace(MemorySpace::DEVICE, &this->dataPtr, this->getSizeInBytes());
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(std::initializer_list<IndexT> sizes)
    : AscendTensor(nullptr, sizes)
{
    this->state = ALLOC_STATE_OWNER;
    allocMemorySpace(MemorySpace::DEVICE, &this->dataPtr, this->getSizeInBytes());
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(AscendMemory &m, const IndexT sizes[Dim], aclrtStream stream)
    : AscendTensor(nullptr, sizes)
{
    this->state = ALLOC_STATE_RESERVATION;

    auto memory = m.getMemory(stream, this->getSizeInBytes());

    this->dataPtr = (T *)memory.get();
    reservation = std::move(memory);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits>::AscendTensor(AscendMemory &m, std::initializer_list<IndexT> sizes,
    aclrtStream stream)
    : AscendTensor(nullptr, sizes)
{
    this->state = ALLOC_STATE_RESERVATION;

    auto memory = m.getMemory(stream, this->getSizeInBytes());

    this->dataPtr = (T *)memory.get();
    reservation = std::move(memory);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits> &AscendTensor<T, Dim, IndexT, PtrTraits>::operator = (
    const AscendTensor<T, Dim, IndexT, PtrTraits> &t)
{
    this->dataPtr = nullptr;
    for (int i = 0; i < Dim; ++i) {
        sizeArray[i] = t.sizeArray[i];
        strideArray[i] = t.strideArray[i];
    }

    this->state = ALLOC_STATE_OWNER;
    allocMemorySpace(MemorySpace::DEVICE, &this->dataPtr, t.getSizeInBytes());
    (void)memcpy_s((void *)this->data(), this->getSizeInBytes(), (void *)t.data(), t.getSizeInBytes());

    return *this;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits> &AscendTensor<T, Dim, IndexT, PtrTraits>::operator = (
    AscendTensor<T, Dim, IndexT, PtrTraits> &&t)
{
    if (this->state == ALLOC_STATE_OWNER && this->dataPtr != nullptr) {
        FreeMemorySpace(MemorySpace::DEVICE, (void *)this->dataPtr);
    }

    dataPtr = t.dataPtr;
    for (int i = 0; i < Dim; ++i) {
        sizeArray[i] = t.sizeArray[i];
        strideArray[i] = t.strideArray[i];
        t.sizeArray[i] = 0;
        t.strideArray[i] = 0;
    }

    t.dataPtr = nullptr;

    this->state = t.state;
    t.state = ALLOC_STATE_NOT_OWNER;
    this->reservation = std::move(t.reservation);

    return *this;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
void AscendTensor<T, Dim, IndexT, PtrTraits>::copyFrom(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream)
{
    // Size must match
    ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

    if (t.numElements() > 0) {
        ASCEND_THROW_IF_NOT(this->data());
        ASCEND_THROW_IF_NOT(t.data());

        ACL_REQUIRE_OK(aclrtMemcpyAsync((void *)this->data(), this->getSizeInBytes(), (void *)t.data(),
            t.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
void AscendTensor<T, Dim, IndexT, PtrTraits>::copyTo(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream)
{
    // Size must match
    ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

    if (this->numElements() > 0) {
        ASCEND_THROW_IF_NOT(data());
        ASCEND_THROW_IF_NOT(t.data());

        ACL_REQUIRE_OK(aclrtMemcpyAsync((void *)t.data(), t.getSizeInBytes(), (void *)this->data(),
            this->getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
void AscendTensor<T, Dim, IndexT, PtrTraits>::copyFromSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t)
{
    // Size must match
    ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

    if (t.numElements() > 0) {
        ASCEND_THROW_IF_NOT(this->data());
        ASCEND_THROW_IF_NOT(t.data());

        (void)memcpy_s((void *)this->data(), this->getSizeInBytes(), (void *)t.data(), t.getSizeInBytes());
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
void AscendTensor<T, Dim, IndexT, PtrTraits>::copyToSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t)
{
    // Size must match
    ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

    if (this->numElements() > 0) {
        ASCEND_THROW_IF_NOT(data());
        ASCEND_THROW_IF_NOT(t.data());

        (void)memcpy_s((void *)t.data(), t.getSizeInBytes(), (void *)this->data(), this->getSizeInBytes());
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
AscendTensor<T, Dim, IndexT, PtrTraits> &AscendTensor<T, Dim, IndexT, PtrTraits>::zero()
{
    if (this->dataPtr != nullptr) {
        auto error = memset_s(this->dataPtr, this->getSizeInBytes(), 0, this->getSizeInBytes());
        ASCEND_THROW_IF_NOT_FMT(error == 0, "failed to memset (error %d)", (int)error);
    }

    return *this;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
template<typename U>
AscendTensor<U, Dim, IndexT, PtrTraits> AscendTensor<T, Dim, IndexT, PtrTraits>::cast()
{
    static_assert(sizeof(U) == sizeof(T), "cast must be to same size object");

    return AscendTensor<U, Dim, IndexT, PtrTraits>(reinterpret_cast<U *>(dataPtr), sizeArray, strideArray);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
template<typename U>
AscendTensor<U, Dim, IndexT, PtrTraits> AscendTensor<T, Dim, IndexT, PtrTraits>::cast() const
{
    static_assert(sizeof(U) == sizeof(T), "cast must be to same size object");

    return AscendTensor<U, Dim, IndexT, PtrTraits>(reinterpret_cast<U *>(dataPtr), sizeArray, strideArray);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
typename AscendTensor<T, Dim, IndexT, PtrTraits>::DataPtrType AscendTensor<T, Dim, IndexT, PtrTraits>::data() const
{
    return dataPtr;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
typename AscendTensor<T, Dim, IndexT, PtrTraits>::DataPtrType AscendTensor<T, Dim, IndexT, PtrTraits>::end() const
{
    return data() + numElements();
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
inline int AscendTensor<T, Dim, IndexT, PtrTraits>::dimNum() const
{
    return NUM_DIM;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
inline const IndexT *AscendTensor<T, Dim, IndexT, PtrTraits>::sizes() const
{
    return sizeArray;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
inline const IndexT *AscendTensor<T, Dim, IndexT, PtrTraits>::strides() const
{
    return strideArray;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
IndexT AscendTensor<T, Dim, IndexT, PtrTraits>::getSize(int i) const
{
    ASCEND_THROW_IF_NOT(i >= 0);
    ASCEND_THROW_IF_NOT(i < Dim);
    return sizeArray[i];
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
IndexT AscendTensor<T, Dim, IndexT, PtrTraits>::getStride(int i) const
{
    ASCEND_THROW_IF_NOT(i >= 0);
    ASCEND_THROW_IF_NOT(i < Dim);
    return strideArray[i];
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
size_t AscendTensor<T, Dim, IndexT, PtrTraits>::numElements() const
{
    size_t size = (size_t)getSize(0);

    for (int i = 1; i < Dim; ++i) {
        size *= (size_t)getSize(i);
    }

    return size;
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
size_t AscendTensor<T, Dim, IndexT, PtrTraits>::getSizeInBytes() const
{
    return sizeof(T) * numElements();
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
void AscendTensor<T, Dim, IndexT, PtrTraits>::initValue(DataType val)
{
    if (this->numElements() > 0) {
        ASCEND_THROW_IF_NOT(this->data());

        std::fill_n(this->dataPtr, this->numElements(), val);
    }
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
template<int SubDim>
AscendTensor<T, SubDim, IndexT, PtrTraits> AscendTensor<T, Dim, IndexT, PtrTraits>::view(DataPtrType data)
{
    ASCEND_THROW_IF_NOT_MSG(SubDim >= 1 && SubDim < Dim, "can only create view of lesser dim");

    IndexT viewSizes[SubDim];
    IndexT viewStrides[SubDim];

    for (int i = 0; i < SubDim; ++i) {
        viewSizes[i] = sizeArray[Dim - SubDim + i];
        viewStrides[i] = strideArray[Dim - SubDim + i];
    }

    return AscendTensor<T, SubDim, IndexT, PtrTraits>(data, viewSizes, viewStrides);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
template<int SubDim>
AscendTensor<T, SubDim, IndexT, PtrTraits> AscendTensor<T, Dim, IndexT, PtrTraits>::view()
{
    return view<SubDim>(data());
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
template<int NewDim>
AscendTensor<T, NewDim, IndexT, PtrTraits> AscendTensor<T, Dim, IndexT, PtrTraits>::view(
    std::initializer_list<IndexT> sizes)
{
    ASCEND_THROW_IF_NOT(sizes.size() == NewDim);

    // The total size of the new view must be the same as the total size
    // of the old view
    size_t curSize = numElements();
    size_t newSize = 1;

    for (auto s : sizes) {
        newSize *= s;
    }

    ASCEND_THROW_IF_NOT(curSize == newSize);
    return AscendTensor<T, NewDim, IndexT, PtrTraits>(data(), sizes);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
detail::SubTensor<AscendTensor<T, Dim, IndexT, PtrTraits>, Dim - 1, PtrTraits>
    AscendTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index)
{
    return detail::SubTensor<AscendTensor::TensorType, Dim - 1, PtrTraits>(
        detail::SubTensor<AscendTensor::TensorType, Dim, PtrTraits>(*this, data())[index]);
}

template<typename T, int Dim, typename IndexT, template<typename U> class PtrTraits>
const detail::SubTensor<AscendTensor<T, Dim, IndexT, PtrTraits>, Dim - 1, PtrTraits>
    AscendTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index) const
{
    return detail::SubTensor<AscendTensor::TensorType, Dim - 1, PtrTraits>(
        detail::SubTensor<AscendTensor::TensorType, Dim, PtrTraits>(const_cast<AscendTensor::TensorType &>(*this),
            data())[index]);
}

namespace detail {
/* ************************************************************ */
/* implementation of SubTensor<TensorType, 0, PtrTraits> */
/* ************************************************************ */
template<typename TensorType, template<typename U> class PtrTraits>
SubTensor<TensorType, 0, PtrTraits>::SubTensor(TensorType &t, typename TensorType::DataPtrType d)
    : tensor(t), dataPtr(d)
{}

template<typename TensorType, template<typename U> class PtrTraits>
SubTensor<TensorType, 0, PtrTraits> SubTensor<TensorType, 0, PtrTraits>::operator = (typename TensorType::DataType val)
{
    *dataPtr = val;
    return *this;
}

template<typename TensorType, template<typename U> class PtrTraits>
SubTensor<TensorType, 0, PtrTraits>::operator typename TensorType::DataType &()
{
    return *dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
SubTensor<TensorType, 0, PtrTraits>::operator const typename TensorType::DataType &() const
{
    return *dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
typename TensorType::DataType *SubTensor<TensorType, 0, PtrTraits>::operator&()
{
    return dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
const typename TensorType::DataType *SubTensor<TensorType, 0, PtrTraits>::operator&() const
{
    return dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
typename TensorType::DataPtrType SubTensor<TensorType, 0, PtrTraits>::data()
{
    return dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
const typename TensorType::DataPtrType SubTensor<TensorType, 0, PtrTraits>::data() const
{
    return dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
typename TensorType::DataType SubTensor<TensorType, 0, PtrTraits>::value()
{
    return *dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
const typename TensorType::DataType SubTensor<TensorType, 0, PtrTraits>::value() const
{
    return *dataPtr;
}

template<typename TensorType, template<typename U> class PtrTraits>
template<typename T>
T &SubTensor<TensorType, 0, PtrTraits>::as()
{
    return *dataAs<T>();
}

template<typename TensorType, template<typename U> class PtrTraits>
template<typename T>
const T &SubTensor<TensorType, 0, PtrTraits>::as() const
{
    return *dataAs<T>();
}

template<typename TensorType, template<typename U> class PtrTraits>
template<typename T>
typename PtrTraits<T>::PtrType SubTensor<TensorType, 0, PtrTraits>::dataAs()
{
    return reinterpret_cast<typename PtrTraits<T>::PtrType>(dataPtr);
}

template<typename TensorType, template<typename U> class PtrTraits>
template<typename T>
typename PtrTraits<const T>::PtrType SubTensor<TensorType, 0, PtrTraits>::dataAs() const
{
    return reinterpret_cast<typename PtrTraits<const T>::PtrType>(dataPtr);
}

/* ************************************************************ */
/* implementation of SubTensor<TensorType, SubDim, PtrTraits> */
/* ************************************************************ */
template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
SubTensor<TensorType, SubDim, PtrTraits>::SubTensor(TensorType &t, typename TensorType::DataPtrType data)
    : tensor(t), dataPtr(data)
{}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
inline SubTensor<TensorType, SubDim - 1, PtrTraits> SubTensor<TensorType, SubDim, PtrTraits>::operator[](
    typename TensorType::IndexType index)
{
    if (SubDim == 1) {
        return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor, dataPtr + index);
    } else {
        return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor,
            dataPtr + index * tensor.getStride(TensorType::NUM_DIM - SubDim));
    }
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
inline const SubTensor<TensorType, SubDim - 1, PtrTraits> SubTensor<TensorType, SubDim, PtrTraits>::operator[](
    typename TensorType::IndexType index) const
{
    if (SubDim == 1) {
        return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor, dataPtr + index);
    } else {
        return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor,
            dataPtr + index * tensor.getStride(TensorType::NUM_DIM - SubDim));
    }
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
typename TensorType::DataType *SubTensor<TensorType, SubDim, PtrTraits>::operator&()
{
    return dataPtr;
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
const typename TensorType::DataType *SubTensor<TensorType, SubDim, PtrTraits>::operator&() const
{
    return dataPtr;
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
typename TensorType::DataPtrType SubTensor<TensorType, SubDim, PtrTraits>::data()
{
    return dataPtr;
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
const typename TensorType::DataPtrType SubTensor<TensorType, SubDim, PtrTraits>::data() const
{
    return dataPtr;
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
template<typename T>
T &SubTensor<TensorType, SubDim, PtrTraits>::as()
{
    return *dataAs<T>();
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
template<typename T>
const T &SubTensor<TensorType, SubDim, PtrTraits>::as() const
{
    return *dataAs<T>();
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
template<typename T>
typename PtrTraits<T>::PtrType SubTensor<TensorType, SubDim, PtrTraits>::dataAs()
{
    return reinterpret_cast<typename PtrTraits<T>::PtrType>(dataPtr);
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
template<typename T>
typename PtrTraits<const T>::PtrType SubTensor<TensorType, SubDim, PtrTraits>::dataAs() const
{
    return reinterpret_cast<typename PtrTraits<const T>::PtrType>(dataPtr);
}

template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
AscendTensor<typename TensorType::DataType, SubDim, typename TensorType::IndexType, PtrTraits>
SubTensor<TensorType, SubDim, PtrTraits>::view()
{
    return tensor.template view<SubDim>(dataPtr);
}
} // namespace detail
} // namespace ascend
#endif // ASCEND_TENSOR_INL_INCLUDED
