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
#include <faiss/ascend/AscendClustering.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <faiss/utils/utils.h>
#include <faiss/utils/random.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <omp.h>
#include <securec.h>

namespace faiss { 
namespace ascend {
namespace {
    const int MAX_POINTS_PER_CENTROID = 60;   // Sample scale of Kmeans||
    const int KMEANS_PARALLEL_ITERATION_NUM = 6;   // Iteration number of Kmeans||
    const int KMEANS_PARALLEL_L = 2;   // Parameter l of Kmeans||
    const int KMEANS_PLUSPLUS_ITERATION_NUM = 32;   // Iteration number of Kmeans++
};

AscendClustering::AscendClustering(int d, int k)
    : Clustering(d, k) {}

AscendClustering::AscendClustering(int d, int k, const faiss::ClusteringParameters &cp)
    : Clustering(d, k, cp) {}

void AscendClustering::train(idx_t nx, const float *x_in, faiss::Index &index)
{
    FAISS_THROW_IF_NOT_FMT((size_t)nx >= k,
                           "Number of training points (%ld) should be at least as large as number of clusters (%ld)",
                           nx, k);
    FAISS_THROW_IF_NOT_MSG(x_in, "x_in can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT(index.d == (int)d, "Index dimension %d not the same as data dimension %d", index.d, (int)d);
    for (size_t i = 0; i < nx * d; i++) {
        FAISS_THROW_IF_NOT_MSG(finite(x_in[i]), "input contains NaN's or Inf's");
    }
    const float *x = x_in;
    idx_t maxLength = MAX_POINTS_PER_CENTROID * k;
    idx_t numX = nx;
    if (nx > maxLength) {
        printf("Sampling a subset of %ld / %ld for Kmeans|| training\n", maxLength, nx);
        std::vector<int> selectXIndex(nx);
        rand_perm(selectXIndex.data(), nx, seed + 15486558L);   // Random seed offset, 15486558L
        float *xSelect = new float[maxLength * d];
        for (int i = 0; i < maxLength; i++) {
            (void)memcpy_s(xSelect + i * d, sizeof(float) * d, x_in + selectXIndex[i] * d, sizeof(float) * d);
        }
        numX = maxLength;
        x = xSelect;
    }
    if (verbose) {
        printf("numX: %ld \n", numX);
    }
    std::vector<float> clusterSet;
    int numC;
    overSample(numX, x, clusterSet, numC, index);
    FAISS_THROW_IF_NOT_MSG(numC > 0, "Error after over-sample: numC <= 0");
    if (verbose) {
        printf("Kmeans|| numC: %d \n", numC);
    }
    float *weights = new float[numC];
    faiss::ScopeDeleter<float> weightsDel(weights);
    calculateWeights(numX, x, numC, &clusterSet[0], weights, index);
    if (x != nullptr && x != x_in) {
        delete [] x;
    }
    kMeansPP(numC, clusterSet, index);
    train(numC, clusterSet.data(), index, weights);
    Clustering::train(nx, x_in, index);
}

void AscendClustering::train(idx_t nx, const float *x, faiss::Index &index, const float *weights)
{
    idx_t *assign = new idx_t[nx];
    faiss::ScopeDeleter<idx_t> assignDel(assign);
    float *dis = new float[nx];
    faiss::ScopeDeleter<float> disDel(dis);
    // support input centroids
    FAISS_THROW_IF_NOT_MSG(centroids.size() % d == 0, "size of provided input centroids not a multiple of dimension");

    // prepare the index
    if (index.ntotal != 0) {
        index.reset();
    }
    if (!index.is_trained) {
        index.train(k, centroids.data());
    }
    index.add(k, centroids.data());
    // k-means iterations
    for (int i = 0; i < KMEANS_PLUSPLUS_ITERATION_NUM; i++) {
        index.search(nx, x, 1, dis, assign);
        computeCentroids(nx, x, assign, weights);
        
        // add centroids to index for the next iteration (or for output)
        index.reset();
        index.train(k, centroids.data());
        index.add(k, centroids.data());
    }
}

void AscendClustering::overSample(idx_t nx, const float *x, std::vector<float> &sample, int &num, faiss::Index &index)
{
    int l = KMEANS_PARALLEL_L * k;
    idx_t firstClusterIndex;
    int64_rand_max(&firstClusterIndex, 1, nx, seed + 15478465L);   // Random seed offset, 15478465L 
    copyVector(sample, x + firstClusterIndex * d, d);
    num = 1;
    std::vector<float> uniformSample(nx * KMEANS_PARALLEL_ITERATION_NUM);
    // Generate uniform sample floats, using random seed offset 15010087L.
    float_rand(uniformSample.data(), nx * KMEANS_PARALLEL_ITERATION_NUM, seed + 15010087L);   
    std::vector<idx_t> assign(nx);
    std::vector<float> dis(nx);
    for (int i = 0; i < KMEANS_PARALLEL_ITERATION_NUM; i++) {    
        if (index.ntotal != 0) {
            index.reset();
        }
        if (!index.is_trained) {
            index.train(num, &sample[0]);
        }
        index.add(num, &sample[0]);
        index.search(nx, x, 1, dis.data(), assign.data());

        float clusteringCost = accumulate(dis.begin(), dis.end(), 0.0f);
        FAISS_THROW_IF_NOT_MSG(clusteringCost > 0, "Error in over-sample: clusteringCost <= 0");
        for (int j = 0; j < nx; j++) {
            float probability = dis[j] / clusteringCost * l;
            if (uniformSample[i * nx + j] < probability) {
                copyVector(sample, x + j * d, d);
                num++;
            }
        }
    }
}

void AscendClustering::calculateWeights(idx_t nx, const float *x,
                                        idx_t nc, const float *centroids, float *weights, faiss::Index &index)
{
    std::vector<idx_t> assign(nx);
    std::vector<float> dis(nx);
    if (index.ntotal != 0) {
        index.reset();
    }
    if (!index.is_trained) {
        index.train(nc, centroids);
    }
    index.add(nc, centroids);
    index.search(nx, x, 1, dis.data(), assign.data());
    std::vector<int> hassign(nc, 0);
    for (int i = 0; i < nx; i++) {
        hassign[assign[i]]++;
    }
    int maxAssign = 0;
    for (int i = 0; i < nc; i++) {
        if (maxAssign < hassign[i]) {
            maxAssign = hassign[i];
        }
    }
    for (int i = 0; i < nc; i++) {
        if (maxAssign != 0) {
            weights[i] = hassign[i] * 1.0f / maxAssign;
        }
    }
}

void AscendClustering::kMeansPP(idx_t nx, std::vector<float> &x, faiss::Index &index)
{
    idx_t firstCentroidIndex;
    int64_rand_max(&firstCentroidIndex, 1, nx, seed + 15846448L);   // Random seed offset, 15846448L
    centroids.resize(d * k);
    std::copy(x.begin() + firstCentroidIndex * d, x.begin() + (firstCentroidIndex + 1) * d, centroids.begin());
    size_t tempK = 1;
    std::vector<float> minDis(nx);
    std::vector<float> roulette(k - 1);
    float_rand(roulette.data(), k - 1, seed + 1);   // Random seed offset, 1
    std::vector<idx_t> assign(nx);
    std::vector<float> dis(nx);
    while (tempK < k) {
        if (index.ntotal != 0) {
            index.reset();
        }
        if (!index.is_trained) {
            index.train(1, &centroids[(tempK - 1) * d]);
        }
        index.add(1, &centroids[(tempK - 1) * d]);
        index.search(nx, &x[0], 1, dis.data(), assign.data());
        for (int i = 0; i < nx; i++) {
            if ((tempK == 1) || (dis[i] < minDis[i])) {
                minDis[i] = dis[i];
            }
        }
        float clusteringCost = accumulate(minDis.begin(), minDis.end(), 0.0f);
        FAISS_THROW_IF_NOT_MSG(clusteringCost > 0, "Error in kMeansPP: clusteringCost <= 0");
        idx_t nextCentroidIndex = 0;
        float accProb = 0;
        while (nextCentroidIndex < nx) {
            accProb += minDis[nextCentroidIndex] / clusteringCost;
            if (accProb > roulette[tempK - 1]) {
                break;
            }
            nextCentroidIndex++;
        }
        nextCentroidIndex = (nextCentroidIndex < nx) ? nextCentroidIndex : (nx - 1);
        std::copy(x.begin() + nextCentroidIndex * d,
                  x.begin() + (nextCentroidIndex + 1) * d,
                  centroids.begin() + tempK * d);
        tempK++;
        if (verbose && tempK % 512 == 0) {   // print process info per 512 times
            printf("Kmeans++, processing: %zu       \r", tempK);
            fflush(stdout);
        }
    }
}

void AscendClustering::copyVector(std::vector<float> &dest, const float *src, int d)
{
    for (int dim = 0; dim < d; dim++) {
        dest.push_back(*(src + dim));
    }
}

void AscendClustering::computeCentroids(idx_t n, const float *x, const idx_t *assign, const float *weights)
{
    std::fill_n(centroids.begin(), d * k, 0.0f);
    std::vector<float> hassign(k, 0.0f);
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();
        nt = (nt < 1) ? 1 : nt;   // at least 1 thread

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;

        for (int i = 0; i < n; i++) {
            size_t ci = (size_t)assign[i];
            FAISS_THROW_IF_NOT_MSG(ci < k, "Error in Kmeans|| computeCentroids.");
            if (ci >= c0 && ci < c1)  {
                float *c = &centroids[ci * d];
                const float *xi = x + i * d;
                float w = weights[i];
                hassign[ci] += w;
                for (size_t j = 0; j < d; j++) {
                    c[j] += xi[j] * w;
                }
            }
        }
    }
#pragma omp parallel for
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / hassign[ci];
        float *c = &centroids[ci * d];
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}
}
}