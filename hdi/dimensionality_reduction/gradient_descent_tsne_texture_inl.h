/*
*
* Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
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


#ifndef GRADIENT_DESCENT_TSNE_TEXTURE_INL
#define GRADIENT_DESCENT_TSNE_TEXTURE_INL

#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"
#include "hdi/utils/math_utils.h"
#include "hdi/utils/log_helper_functions.h"

namespace hdi {
  namespace dr {

    template <typename sparse_scalar_matrix_type>
    GradientDescentTSNETexture<sparse_scalar_matrix_type>::GradientDescentTSNETexture() :
      _embedding(nullptr),
      _embedding_container(nullptr),
      _initialized(false),
      _exaggeration_baseline(1),
      _P(),
      _Q(),
      _params(),
      _iteration(0),
      _logger(nullptr),
#ifndef __APPLE__
      _gpgpu_compute_tsne(),
      _gpgpu_type(AUTO_DETECT),
#endif
      _gpgpu_raster_tsne()
    {
    }
    template GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::GradientDescentTSNETexture();
    template GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::GradientDescentTSNETexture();

#ifndef __APPLE__
    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::setType(GpgpuSneType tsne_type) {
      if (tsne_type == AUTO_DETECT)
      {
        //resolve the optimal type to use based on the available OpenGL version
        if (GLAD_GL_VERSION_4_3)
        {
          _gpgpu_type = COMPUTE_SHADER;
        }
        else if (GLAD_GL_VERSION_3_3)
        {
          std::cout << "Compute shaders not available, using rasterization fallback" << std::endl;
          _gpgpu_type = RASTER;
        }
      }
      else
        _gpgpu_type = tsne_type;
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::setType(GpgpuSneType);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::setType(GpgpuSneType);
#endif

    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::reset() {
      _initialized = false;
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::reset();
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::reset();

    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::clear() {
      _embedding->clear();
      _initialized = false;
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::clear();
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::clear();

    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::getEmbeddingPosition(scalar_vector_type& embedding_position, data_handle_type handle) const {
      if (!_initialized) {
        throw std::logic_error("Algorithm must be initialized before ");
      }
      embedding_position.resize(_params._embedding_dimensionality);
      for (int i = 0; i < _params._embedding_dimensionality; ++i) {
        (*_embedding_container)[i] = (*_embedding_container)[static_cast<size_t>(handle) * _params._embedding_dimensionality + i];
      }
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::getEmbeddingPosition(scalar_vector_type&, data_handle_type) const;
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::getEmbeddingPosition(scalar_vector_type&, data_handle_type) const;


    /////////////////////////////////////////////////////////////////////////


    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::initialize(const sparse_scalar_matrix_type& probabilities, data::Embedding<scalar_type>* embedding, TsneParameters params) {
      utils::secureLog(_logger, "Initializing tSNE...");
      {//Aux data
        _params = params;
        const size_t size = probabilities.size();

        _embedding = embedding;
        _embedding_container = &(embedding->getContainer());
        _embedding->resize(static_cast<size_t>(_params._embedding_dimensionality), size);
        _P.clear();
        _P.resize(size);
      }

      utils::secureLogValue(_logger, "Number of data points", _P.size());

      computeHighDimensionalDistribution(probabilities);
      if (!_params._presetEmbedding) {
        initializeEmbeddingPosition(_params._seed, _params._rngRange);
      }

#ifndef __APPLE__
      if (_gpgpu_type == AUTO_DETECT)
        setType(AUTO_DETECT); // resolves whether to use Compute Shader or Raster version
      if (_gpgpu_type == COMPUTE_SHADER)
        _gpgpu_compute_tsne.initialize(_embedding, _params, _P);
      else// (_tsne_type == RASTER)
        _gpgpu_raster_tsne.initialize(_embedding, _params, _P);
#else
      _gpgpu_raster_tsne.initialize(_embedding, _params, _P);
#endif

      _iteration = 0;

      _initialized = true;
      utils::secureLog(_logger, "Initialization complete!");
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::initialize(const std::vector<hdi::data::SparseVec<uint32_t, float>>&, data::Embedding<scalar_type>*, TsneParameters);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::initialize(const std::vector<hdi::data::MapMemEff<uint32_t, float>>&, data::Embedding<scalar_type>*, TsneParameters);

    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::initializeWithJointProbabilityDistribution(const sparse_scalar_matrix_type& distribution, data::Embedding<scalar_type>* embedding, TsneParameters params) {
      utils::secureLog(_logger, "Initializing tSNE with a user-defined joint-probability distribution...");
      {//Aux data
        _params = params;
        const size_t size = distribution.size();

        _embedding = embedding;
        _embedding_container = &(embedding->getContainer());
        _embedding->resize(_params._embedding_dimensionality, size);
        _P.resize(size);
      }

      utils::secureLogValue(_logger, "Number of data points", _P.size());

      _P = distribution;
      if (!_params._presetEmbedding) {
        initializeEmbeddingPosition(_params._seed, _params._rngRange);
      }

#ifndef __APPLE__
      if (_gpgpu_type == AUTO_DETECT)
        setType(AUTO_DETECT); // resolves whether to use Compute Shader or Raster version
      if (_gpgpu_type == COMPUTE_SHADER)
        _gpgpu_compute_tsne.initialize(_embedding, _params, _P);
      else// (_tsne_type == RASTER)
        _gpgpu_raster_tsne.initialize(_embedding, _params, _P);
#else
      _gpgpu_raster_tsne.initialize(_embedding, _params, _P);
#endif

      _iteration = 0;

      _initialized = true;
      utils::secureLog(_logger, "Initialization complete!");
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::initializeWithJointProbabilityDistribution(const std::vector<hdi::data::SparseVec<uint32_t, float>>&, data::Embedding<scalar_type>*, TsneParameters);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::initializeWithJointProbabilityDistribution(const std::vector<hdi::data::MapMemEff<uint32_t, float>>&, data::Embedding<scalar_type>*, TsneParameters);

    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::updateParams(TsneParameters params) {
      if (!_initialized) {
        throw std::runtime_error("GradientDescentTSNETexture must be initialized before updating the tsne parameters");
      }
      _params = params;
#ifndef __APPLE__
      _gpgpu_compute_tsne.updateParams(params);
#else

      _gpgpu_raster_tsne.updateParams(params);
#endif
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::updateParams(TsneParameters);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::updateParams(TsneParameters);


    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::computeHighDimensionalDistribution(const sparse_scalar_matrix_type& probabilities) {
      utils::secureLog(_logger, "Computing high-dimensional joint probability distribution...");

      const size_t n = getNumberOfDataPoints();

      if constexpr (std::is_same_v<sparse_scalar_matrix_type, std::vector<hdi::data::SparseVec<uint32_t, float>>>)
      {
        for (size_t j = 0; j < n; ++j)
          _P[j].resize(n);

        for (size_t j = 0; j < n; ++j) {
          for (Eigen::SparseVector<float>::InnerIterator it(probabilities[j].memory()); it; ++it) {
            scalar_type v0 = it.value();
            scalar_type v1 = 0.;
            if (probabilities[it.index()].coeff(j) != 0.0)
              v1 = probabilities[it.index()].coeff(j);

            const auto jointProb = static_cast<scalar_type>((v0 + v1) * 0.5);

            _P[j][it.index()] = jointProb;
            _P[it.index()][j] = jointProb;
          }
        }
      }
      else // MapMemEff
      {
        for (uint32_t j = 0; j < n; ++j) {
          for (auto& elem : probabilities[j]) {
            const uint32_t i = elem.first;
            scalar_type new_val = (probabilities[j][i] + probabilities[i][j]) * 0.5f;
            _P[j][i] = new_val;
            _P[i][j] = new_val;
          }
        }

      } // MapMemEff
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::computeHighDimensionalDistribution(const std::vector<hdi::data::SparseVec<uint32_t, float>>&);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::computeHighDimensionalDistribution(const std::vector<hdi::data::MapMemEff<uint32_t, float>>&);


    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::initializeEmbeddingPosition(int seed, double multiplier) {
      utils::secureLog(_logger, "Initializing the embedding...");

      if (seed < 0) {
        std::srand(static_cast<unsigned int>(time(NULL)));
      }
      else {
        std::srand(seed);
      }

      for (size_t i = 0; i < _embedding->numDataPoints(); ++i) {
        double x(0.);
        double y(0.);
        double radius(0.);
        do {
          x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
          y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
          radius = (x * x) + (y * y);
        } while ((radius >= 1.0) || (radius == 0.0));

        radius = sqrt(-2 * log(radius) / radius);
        x *= radius * multiplier;
        y *= radius * multiplier;
        _embedding->dataAt(i, 0) = x;
        _embedding->dataAt(i, 1) = y;
      }
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::initializeEmbeddingPosition(int, double);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::initializeEmbeddingPosition(int, double);

    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::doAnIteration(double mult) {
      if (!_initialized) {
        throw std::logic_error("Cannot compute a gradient descent iteration on unitialized data");
      }

      if (_iteration == _params._mom_switching_iter) {
        utils::secureLog(_logger, "Switch to final momentum...");
      }
      if (_iteration == _params._remove_exaggeration_iter) {
        utils::secureLog(_logger, "Remove exaggeration...");
      }

      doAnIterationImpl(mult);
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::doAnIteration(double);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::doAnIteration(double);

    template <typename sparse_scalar_matrix_type>
    double GradientDescentTSNETexture<sparse_scalar_matrix_type>::exaggerationFactor() {
      scalar_type exaggeration = _exaggeration_baseline;

      if (_iteration <= _params._remove_exaggeration_iter) {
        exaggeration = _params._exaggeration_factor;
      }
      else if (_iteration <= (_params._remove_exaggeration_iter + _params._exponential_decay_iter)) {
        //double decay = std::exp(-scalar_type(_iteration-_params._remove_exaggeration_iter)/30.);
        double decay = 1. - double(_iteration - _params._remove_exaggeration_iter) / _params._exponential_decay_iter;
        exaggeration = _exaggeration_baseline + (_params._exaggeration_factor - _exaggeration_baseline)*decay;
        //utils::secureLogValue(_logger,"Exaggeration decay...",exaggeration);
      }

      return exaggeration;
    }
    template double GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::exaggerationFactor();
    template double GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::exaggerationFactor();

    template <typename sparse_scalar_matrix_type>
    void GradientDescentTSNETexture<sparse_scalar_matrix_type>::doAnIterationImpl(double mult) {
      // Compute gradient of the KL function using a compute shader approach
#ifndef __APPLE__
      if (_gpgpu_type == COMPUTE_SHADER)
        _gpgpu_compute_tsne.compute(_embedding, exaggerationFactor(), static_cast<float>(_iteration), mult);
      else
        _gpgpu_raster_tsne.compute(_embedding, exaggerationFactor(), static_cast<float>(_iteration), mult);
#else
      _gpgpu_raster_tsne.compute(_embedding, exaggerationFactor(), _iteration, mult);
#endif
      ++_iteration;
    }
    template void GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::doAnIterationImpl(double);
    template void GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::doAnIterationImpl(double);

    template <typename sparse_scalar_matrix_type>
    double GradientDescentTSNETexture<sparse_scalar_matrix_type>::computeKullbackLeiblerDivergence(bool computeQ) {
      const size_t n = _embedding->numDataPoints();

      if(computeQ)
        _Q.resize(n * n);

      double sum_Q = 0;
      for (size_t j = 0; j < n; ++j) {

        if (computeQ)
          _Q[j*n + j] = 0;

        for (size_t i = j + 1; i < n; ++i) {
          const double euclidean_dist_sq(
            utils::euclideanDistanceSquared<float>(
              _embedding->getContainer().begin() + j * _params._embedding_dimensionality,
              _embedding->getContainer().begin() + (j + 1)*_params._embedding_dimensionality,
              _embedding->getContainer().begin() + i * _params._embedding_dimensionality,
              _embedding->getContainer().begin() + (i + 1)*_params._embedding_dimensionality
              )
          );
          const double v = 1. / (1. + euclidean_dist_sq);

          if (computeQ)
          {
            _Q[j*n + i] = static_cast<float>(v);
            _Q[i*n + j] = static_cast<float>(v);
          }
          sum_Q += v * 2;
        }
      }

      double kl = 0;

      for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<sparse_scalar_matrix_type, std::vector<hdi::data::SparseVec<uint32_t, float>>>)
        {
          for (Eigen::SparseVector<float>::InnerIterator it(_P[i].memory()); it; ++it) {
            auto j = it.index();

            // Calculate Qij
            const double euclidean_dist_sq(
              utils::euclideanDistanceSquared<float>(
                _embedding->getContainer().begin() + j * _params._embedding_dimensionality,
                _embedding->getContainer().begin() + (j + 1) * _params._embedding_dimensionality,
                _embedding->getContainer().begin() + i * _params._embedding_dimensionality,
                _embedding->getContainer().begin() + (i + 1) * _params._embedding_dimensionality
              )
            );
            const double v = 1. / (1. + euclidean_dist_sq);

            double p = it.value() / (2 * n);
            float klc = p * std::log(p / (v / sum_Q));
            //if (klc > 0.00001)
            //{
            //  std::cout << "KLC: " << klc << " i: " << i << "neighbour: " << neighbour_id << std::endl;
            //}

            kl += klc;
          }
        }
        else // MapMemEff
        {
          for (const auto& pij : _P[i]) {
            auto j = pij.first;

            // Calculate Qij
            const double euclidean_dist_sq(
              utils::euclideanDistanceSquared<float>(
                _embedding->getContainer().begin() + j * _params._embedding_dimensionality,
                _embedding->getContainer().begin() + (j + 1) * _params._embedding_dimensionality,
                _embedding->getContainer().begin() + i * _params._embedding_dimensionality,
                _embedding->getContainer().begin() + (i + 1) * _params._embedding_dimensionality
              )
            );
            const double v = 1. / (1. + euclidean_dist_sq);

            double p = pij.second / (2 * n);
            float klc = p * std::log(p / (v / sum_Q));
            //if (klc > 0.00001)
            //{
            //  std::cout << "KLC: " << klc << " i: " << i << "neighbour: " << neighbour_id << std::endl;
            //}

            kl += klc;
          }
        }
      }
      return kl;
    }
    template double GradientDescentTSNETexture<std::vector<hdi::data::SparseVec<uint32_t, float>>>::computeKullbackLeiblerDivergence(bool);
    template double GradientDescentTSNETexture<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::computeKullbackLeiblerDivergence(bool);

  }
}
#endif
