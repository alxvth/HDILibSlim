/*
 *
 * Copyright (c) 2019, BIOVAULT (Leiden University Medical Center, Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY BIOVAULT ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include "knn_utils.h"
#include "hierarchical_sne.h"
#include "hd_joint_probability_generator.h"

#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"

#include "hdi/data/map_mem_eff.h"
#include "hdi/data/sparse_mat.h"

#include "hnswlib/hnswlib.h"
#pragma warning( push )
#pragma warning( disable : 4477 )
#include "annoylib.h"
#pragma warning( pop )
#include "kissrandom.h"

#ifndef KNN_H_INL
#define KNN_H_INL

namespace hdi{
  namespace dr {


    template <typename scalar_type, typename integer_type, typename stats_type>
    void computeHighDimensionalDistances(const scalar_type* high_dimensional_data, size_t num_dim, size_t num_dps, knn_params params, std::vector<scalar_type>& distances_squared, std::vector<integer_type>& neighborhood_indices, stats_type* statistics, utils::AbstractLog* logger) {
      const auto nn = params.num_neigh;
      distances_squared.resize(num_dps * nn);
      neighborhood_indices.resize(num_dps * nn);

      if (params.lib == hdi::dr::KNN_HNSW)
      {
        if(logger) utils::secureLog(logger, "Computing the neighborhood graph with HNSW Lib...");

        hnswlib::SpaceInterface<float>* space = NULL;
        switch (params.metric) {
        case hdi::dr::KNN_METRIC_EUCLIDEAN:
          if (logger) utils::secureLog(logger, "Computing approximated knn with HNSW using Euclidean distances ...");
          space = new hnswlib::L2Space(num_dim);
          break;
        case hdi::dr::KNN_METRIC_INNER_PRODUCT:
          if (logger) utils::secureLog(logger, "Computing approximated knn with HNSW using inner product (dot) distances ...");
          space = new hnswlib::InnerProductSpace(num_dim);
          break;
        default:
          if (logger) utils::secureLog(logger, "Computing approximated knn with HNSW using Euclidean distances ...");
          space = new hnswlib::L2Space(num_dim);
          break;
        }

        if (logger) utils::secureLog(logger, "\tBuilding the trees...");
        hnswlib::HierarchicalNSW<scalar_type> appr_alg(space, num_dps, params.hnsw_M, params.ef_construction, 0);
        {
          if (statistics) utils::ScopedTimer<scalar_type, utils::Seconds> timer(statistics->_init_knn_time);
          appr_alg.addPoint(static_cast<const void*>(high_dimensional_data), (std::size_t)0);
          unsigned num_threads = std::thread::hardware_concurrency();
          hnswlib::ParallelFor(1, num_dps, num_threads, [&](size_t i, size_t threadId) {
            appr_alg.addPoint(static_cast<const void*>(high_dimensional_data + (i * num_dim)), (hnswlib::labeltype)i);
            });
        }

        appr_alg.setEf(params.ef_construction); // set search ef parameter

        if (logger) utils::secureLog(logger, "\tAKNN queries...");
        {
          if (statistics) utils::ScopedTimer<scalar_type, utils::Seconds> timer(statistics->_comp_knn_time);
#pragma omp parallel for
          for (int i = 0; i < num_dps; ++i)
          {
            auto top_candidates = appr_alg.searchKnn(high_dimensional_data + (i * num_dim), (hnswlib::labeltype)nn);

            while (top_candidates.size() > nn) {
              if (logger) utils::secureLog(logger, "\tI SHOULD NEVER BE HERE - DELETE THIS LOOP...");

              top_candidates.pop();
            }

            scalar_type* distances = distances_squared.data() + (i * nn);
            int* indices = neighborhood_indices.data() + (i * nn);
            int j = 0;
            assert(top_candidates.size() == nn);
            while (top_candidates.size() > 0)
            {
              auto rez = top_candidates.top();
              distances[nn - j - 1] = rez.first;
              indices[nn - j - 1] = rez.second;
              top_candidates.pop();
              ++j;
            }
          }
        }
      }
      else // params.lib == hdi::dr::KNN_ANNOY
      {
        if (logger) utils::secureLog(logger, "Computing approximated knn with Annoy...");

        int search_k = nn * params.num_trees;

        Annoy::AnnoyIndexInterface<int32_t, double>* tree = nullptr;
        switch (params.metric) {
        case hdi::dr::KNN_METRIC_EUCLIDEAN:
          if (logger) utils::secureLog(logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new Annoy::AnnoyIndex<int32_t, double, Annoy::Euclidean, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        case hdi::dr::KNN_METRIC_COSINE:
          if (logger) utils::secureLog(logger, "Computing approximated knn with Annoy using Cosine distances ...");
          tree = new Annoy::AnnoyIndex<int32_t, double, Annoy::Angular, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        case hdi::dr::KNN_METRIC_MANHATTAN:
          if (logger) utils::secureLog(logger, "Computing approximated knn with Annoy using Manhattan distances ...");
          tree = new Annoy::AnnoyIndex<int32_t, double, Annoy::Manhattan, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        //case hdi::dr::KNN_METRIC_HAMMING:
        //  if (logger.has_value())  utils::secureLog(logger.value(), "Computing approximated knn with Annoy using Euclidean distances ...");
        //  tree = new Annoy::AnnoyIndex<int32_t, double, Annoy::Hamming, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
        //  break;
        case hdi::dr::KNN_METRIC_INNER_PRODUCT:
          if (logger) utils::secureLog(logger, "Computing approximated knn with Annoy using Inner Product (Dot) distances ...");
          tree = new Annoy::AnnoyIndex<int32_t, double, Annoy::DotProduct, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        default:
          if (logger) utils::secureLog(logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new Annoy::AnnoyIndex<int32_t, double, Annoy::Euclidean, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        }

        if (logger) utils::secureLog(logger, "\tBuilding the trees...");
        {
          if (statistics) utils::ScopedTimer<scalar_type, utils::Seconds> timer(statistics->_init_knn_time);

          for (int i = 0; i < num_dps; ++i) {
            double* vec = new double[num_dim];
            for (int z = 0; z < num_dim; ++z) {
              vec[z] = high_dimensional_data[i * num_dim + z];
            }
            tree->add_item(i, vec);
          }
          tree->build(params.num_trees);

          // Sample check if it returns enough neighbors
          std::vector<int> closest;
          std::vector<double> closest_distances;
          for (int n = 0; n < 100; n++) {
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);
            unsigned int neighbors_count = closest.size();
            if (neighbors_count < nn) {
              printf("Requesting %zd neighbors, but ANNOY returned only %u. Please increase search_k\n", nn, neighbors_count);
              return;
            }
          }
        }

        if (logger) utils::secureLog(logger, "\tAKNN queries...");
        {
          if (statistics) utils::ScopedTimer<scalar_type, utils::Seconds> timer(statistics->_comp_knn_time);

#pragma omp parallel for
          for (int n = 0; n < num_dps; n++)
          {
            // Find nearest neighbors
            std::vector<int> closest;
            std::vector<double> closest_distances;
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);

            // Copy current row
            for (unsigned int m = 0; m < nn; m++) {
              neighborhood_indices[n * nn + m] = closest[m];
              distances_squared[n * nn + m] = closest_distances[m] * closest_distances[m];
            }
          }
        }
        delete tree;
      }

    }
    template void computeHighDimensionalDistances<float, int, HierarchicalSNE<float, std::vector<std::map<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, HierarchicalSNE<float, std::vector<std::map<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
    template void computeHighDimensionalDistances<float, int, HierarchicalSNE<float, std::vector<std::unordered_map<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, HierarchicalSNE<float, std::vector<std::unordered_map<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
    template void computeHighDimensionalDistances<float, int, HierarchicalSNE<float, std::vector<hdi::data::MapMemEff<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, HierarchicalSNE<float, std::vector<hdi::data::MapMemEff<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
    template void computeHighDimensionalDistances<float, int, HierarchicalSNE<float, std::vector<hdi::data::SparseVec<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, HierarchicalSNE<float, std::vector<hdi::data::SparseVec<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
    template void computeHighDimensionalDistances<float, int, HDJointProbabilityGenerator<float, std::vector<std::map<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, class HDJointProbabilityGenerator<float, std::vector<std::map<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
    template void computeHighDimensionalDistances<float, int, HDJointProbabilityGenerator<float, std::vector<std::unordered_map<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, class HDJointProbabilityGenerator<float, std::vector<std::unordered_map<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
    template void computeHighDimensionalDistances<float, int, HDJointProbabilityGenerator<float, std::vector<hdi::data::MapMemEff<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, class HDJointProbabilityGenerator<float, std::vector<hdi::data::MapMemEff<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
    template void computeHighDimensionalDistances<float, int, HDJointProbabilityGenerator<float, std::vector<hdi::data::SparseVec<unsigned int, float>>>::Statistics>(const float*, size_t, size_t, knn_params, std::vector<float>&, std::vector<int>&, class HDJointProbabilityGenerator<float, std::vector<hdi::data::SparseVec<unsigned int, float>>>::Statistics*, utils::AbstractLog*);
}
}

#endif // KNN_H
