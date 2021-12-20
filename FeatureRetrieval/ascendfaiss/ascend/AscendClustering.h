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
#ifndef ASCEND_CLUSTERING_INCLUDED
#define ASCEND_CLUSTERING_INCLUDED
#include <faiss/Clustering.h>

namespace faiss {
namespace ascend {
/*
 * Implementation of Kmeans||, see paper on https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
 */
class AscendClustering : public faiss::Clustering {
public:
    AscendClustering(int d, int k);
    AscendClustering(int d, int k, const faiss::ClusteringParameters &cp);
    void train(idx_t nx, const float *x_in, faiss::Index &index) override;
    void train(idx_t nx, const float *x, faiss::Index &index, const float *weights);
    virtual ~AscendClustering() {}

private:
    void overSample(idx_t nx, const float *x, std::vector<float> &sample, int &num, faiss::Index &index);
    void calculateWeights(idx_t nx, const float *x,
                          idx_t nc, const float *centroids, float *weights, faiss::Index &index);
    void kMeansPP(idx_t nx, std::vector<float> &x, faiss::Index &index);
    void copyVector(std::vector<float> &dest, const float *src, int d);
    void computeCentroids(idx_t n, const float *x, const idx_t *assign, const float *weights);
};
}
}

#endif
