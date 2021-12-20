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

#include <faiss/ascend/AscendCloner.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/ascend/AscendIndexIVFPQ.h>
#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendIndexIVFFlat.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendIndexPreTransform.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/AscendIndexInt8IVFFlat.h>

namespace faiss {
namespace ascend {
/**********************************************************
 * Cloning to CPU
 **********************************************************/
void ToCPUCloner::merge_index(Index *dst, Index *src, bool successive_ids)
{
    if (auto ifl = dynamic_cast<IndexFlat *>(dst)) {
        auto ifl2 = dynamic_cast<const IndexFlat *>(src);
        FAISS_ASSERT(ifl2);
        FAISS_ASSERT(successive_ids);
        ifl->add(ifl2->ntotal, ifl2->xb.data());
    } else if (auto ipq = dynamic_cast<IndexIVFPQ *>(dst)) {
        auto ipq2 = dynamic_cast<IndexIVFPQ *>(src);
        FAISS_ASSERT(ipq2);
        ipq->merge_from(*ipq2, successive_ids ? ipq->ntotal : 0);
    } else if (auto isq = dynamic_cast<IndexIVFScalarQuantizer *>(dst)) {
        auto isq2 = dynamic_cast<IndexIVFScalarQuantizer *>(src);
        FAISS_ASSERT(isq2);
        isq->merge_from(*isq2, successive_ids ? isq->ntotal : 0);
    } else {
        FAISS_THROW_MSG("merging not supported for this type of Index in Ascend's ToCPUCloner");
    }
}

Index *ToCPUCloner::clone_Index(const Index *index)
{
    if (auto ipq = dynamic_cast<const AscendIndexIVFPQ *>(index)) {
        IndexIVFPQ *res = new IndexIVFPQ();
        ipq->copyTo(res);
        return res;
    } else if (auto flat = dynamic_cast<const AscendIndexFlat *>(index)) {
        IndexFlat *res = new IndexFlat();
        flat->copyTo(res);
        return res;
    } else if (auto iflat = dynamic_cast<const AscendIndexIVFFlat *>(index)) {
        IndexIVFFlat *res = new IndexIVFFlat();
        iflat->copyTo(res);
        return res;
    } else if (auto isq = dynamic_cast<const AscendIndexIVFSQ *>(index)) {
        IndexIVFScalarQuantizer *res = new IndexIVFScalarQuantizer();
        isq->copyTo(res);
        return res;
    } else if (auto pt = dynamic_cast<const AscendIndexPreTransform *>(index)) {
        IndexPreTransform *res = new IndexPreTransform();
        pt->copyTo(res);
        return res;
    } else if (auto sq = dynamic_cast<const AscendIndexSQ *>(index)) {
        IndexScalarQuantizer *res = new IndexScalarQuantizer();
        sq->copyTo(res);
        return res;
    } else {
        FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
        return nullptr;
    }
}

Index *ToCPUCloner::clone_IndexInt8(const AscendIndexInt8 *index)
{
    if (auto int8flat = dynamic_cast<const AscendIndexInt8Flat *>(index)) {
        IndexScalarQuantizer *res = new IndexScalarQuantizer();
        int8flat->copyTo(res);
        return res;
    } else if (auto ivfint8flat = dynamic_cast<const AscendIndexInt8IVFFlat *>(index)) {
        IndexIVFScalarQuantizer *res = new IndexIVFScalarQuantizer();
        ivfint8flat->copyTo(res);
        return res;
    } else {
        FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
        return nullptr;
    }
}

faiss::Index *index_ascend_to_cpu(const faiss::Index *ascendIndex)
{
    ToCPUCloner cl;
    return cl.clone_Index(ascendIndex);
}

faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *ascendIndex)
{
    ToCPUCloner cl;
    return cl.clone_IndexInt8(ascendIndex);
}

/*
 * Cloning to Ascend
 */
ToAscendCloner::ToAscendCloner(std::initializer_list<int> devs, const AscendClonerOptions &options)
    : AscendClonerOptions(options), devices(devs)
{
    FAISS_THROW_IF_NOT_MSG(devs.size() != 0, "device list can not be empty!");
}

ToAscendCloner::ToAscendCloner(std::vector<int> devs, const AscendClonerOptions &options)
    : AscendClonerOptions(options), devices(devs)
{
    FAISS_THROW_IF_NOT_MSG(devs.size() != 0, "device list can not be empty!");
}

Index *ToAscendCloner::clone_Index(const Index *index)
{
    if (auto ipq = dynamic_cast<const IndexIVFPQ *>(index)) {
        if (verbose) {
            printf("IndexIVFPQ size %ld -> AscendIndexIVFPQ, reserverVecs=%ld\n", ipq->ntotal, reserveVecs);
        }

        AscendIndexIVFPQConfig config(devices, resourceSize);
        return new AscendIndexIVFPQ(ipq, config);
    } else if (auto flat = dynamic_cast<const IndexFlat *>(index)) {
        if (verbose) {
            printf("IndexFlat size %ld -> AscendIndexFlat, reserverVecs=%ld\n", flat->ntotal, reserveVecs);
        }

        AscendIndexFlatConfig config(devices, resourceSize);
        return new AscendIndexFlat(flat, config);
    } else if (auto iflat = dynamic_cast<const IndexIVFFlat *>(index)) {
        if (verbose) {
            printf("IndexIVFFlat size %ld -> AscendIndexIVFFlat, reserverVecs=%ld\n", iflat->ntotal, reserveVecs);
        }

        AscendIndexIVFFlatConfig config(devices, resourceSize);
        return new AscendIndexIVFFlat(iflat, config);
    } else if (auto isq = dynamic_cast<const IndexIVFScalarQuantizer *>(index)) {
        if (verbose) {
            printf("IndexIVFScalarQuantizer size %ld -> AscendIndexIVFSQ, reserverVecs=%ld\n", 
                isq->ntotal, reserveVecs);
        }

        AscendIndexIVFSQConfig config(devices, resourceSize);
        return new AscendIndexIVFSQ(isq, config);
    } else if (auto sq = dynamic_cast<const IndexScalarQuantizer *>(index)) {
        if (verbose) {
            printf("IndexScalarQuantizer size %ld -> AscendIndexSQ, reserverVecs=%ld\n", sq->ntotal, reserveVecs);
        }

        AscendIndexSQConfig config(devices, resourceSize);
        return new AscendIndexSQ(sq, config);
    } else if (auto pt = dynamic_cast<const IndexPreTransform *>(index)) {
        AscendIndexConfig config(devices, resourceSize);
        return new AscendIndexPreTransform(pt, config);
    } else {
        FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
        return nullptr;
    }
}

AscendIndexInt8 *ToAscendCloner::clone_IndexInt8(const Index *index)
{
    if (auto isq = dynamic_cast<const IndexIVFScalarQuantizer *>(index)) {
        if (verbose) {
            printf("IndexIVFScalarQuantizer size %ld -> AscendIndexInt8IVFFlat, reserverVecs=%ld\n", 
                isq->ntotal, reserveVecs);
        }

        AscendIndexInt8IVFFlatConfig config;
        config.deviceList = devices;
        config.resourceSize = resourceSize;
        return new AscendIndexInt8IVFFlat(isq, config);
    } else if (auto sq = dynamic_cast<const IndexScalarQuantizer *>(index)) {
        if (verbose) {
            printf("IndexScalarQuantizer size %ld -> AscendIndexInt8Flat, reserverVecs=%ld\n", sq->ntotal, reserveVecs);
        }

        AscendIndexInt8FlatConfig config;
        config.deviceList = devices;
        config.resourceSize = resourceSize;
        return new AscendIndexInt8Flat(sq, config);
    } else {
        FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
        return nullptr;
    }
}

faiss::Index *index_cpu_to_ascend(
    std::initializer_list<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_Index(index);
}

faiss::Index *index_cpu_to_ascend(
    std::vector<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_Index(index);
}

AscendIndexInt8 *index_int8_cpu_to_ascend(
    std::initializer_list<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_IndexInt8(index);
}

AscendIndexInt8 *index_int8_cpu_to_ascend(
    std::vector<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_IndexInt8(index);
}
}  // namespace ascend
}  // namespace faiss
