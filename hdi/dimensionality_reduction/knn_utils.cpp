#include "knn_utils_inl.h"
#include <stdexcept>

#include "hierarchical_sne.h"

namespace hdi {
  namespace dr {
    int HierarchicalSNE_NrOfKnnAlgorithms()
    {
      int numSupported = 1;
#ifdef HNSWLIB_SUPPORTED
      numSupported++;
#endif

#ifdef __USE_ANNOY__
      numSupported++;
#endif
      return numSupported;
    }

    std::map<std::string, int> supported_knn_libraries()
    {
      std::map<std::string, int> result;
#ifdef HNSWLIB_SUPPORTED
      result["HNSW"] = hdi::dr::KNN_HNSW;
#endif
#ifdef __USE_ANNOY__
      result["ANNOY"] = hdi::dr::KNN_ANNOY;
#endif

      return result;
    }

    std::map<std::string, int> supported_knn_library_distance_metrics(int knn_lib)
    {
      std::map<std::string, int> result;
      result["Euclidean"] = hdi::dr::KNN_METRIC_EUCLIDEAN;

      switch (knn_lib)
      {
      case hdi::dr::KNN_HNSW: {
        result["Inner Product (Dot)"] = hdi::dr::KNN_METRIC_INNER_PRODUCT;
        return result;
      }
      case hdi::dr::KNN_ANNOY: {
        result["Cosine"] = hdi::dr::KNN_METRIC_COSINE;
        result["Manhattan"] = hdi::dr::KNN_METRIC_MANHATTAN;
        result["Inner Product (Dot)"] = hdi::dr::KNN_METRIC_INNER_PRODUCT;
        return result;
      }

      default: {
        throw std::out_of_range("knn_lib value out of range");
      }
      }
    }

  }
}



