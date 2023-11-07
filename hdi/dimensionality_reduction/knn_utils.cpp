#include "knn_utils_inl.h"
#include <stdexcept>

namespace hdi {
  namespace dr {

    std::map<std::string, int> supported_knn_libraries()
    {
      std::map<std::string, int> result;
      result["HNSW"] = hdi::dr::KNN_HNSW;
      result["ANNOY"] = hdi::dr::KNN_ANNOY;

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
