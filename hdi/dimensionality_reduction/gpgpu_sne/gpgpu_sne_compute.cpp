#ifndef __APPLE__

#include "gpgpu_sne_compute.h"
#include "compute_shaders.glsl"

#include <vector>
#include <limits>
#include <iostream>
#include <type_traits>
#include <cmath> // for sqrt

namespace hdi {
  namespace dr {

    enum BufferType
    {
      POSITION,
      INTERP_FIELDS,
      SUM_Q,
      NEIGHBOUR,
      PROBABILITIES,
      INDEX,
      GRADIENTS,
      PREV_GRADIENTS,
      GAIN,
      BOUNDS
    };

    // Linearized sparse neighbourhood matrix
    struct LinearProbabilityMatrix
    {
      std::vector<uint32_t> neighbours;
      std::vector<float> probabilities;
      std::vector<int> indices;
    };


    template <typename sparse_scalar_matrix_type>
    GpgpuSneCompute<sparse_scalar_matrix_type>::GpgpuSneCompute() :
      _initialized(false),
      _adaptive_resolution(true),
      _resolutionScaling(PIXEL_RATIO)
    {

    }
    template GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::GpgpuSneCompute();
    template GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::GpgpuSneCompute();

    template <typename sparse_scalar_matrix_type>
    typename GpgpuSneCompute<sparse_scalar_matrix_type>::Bounds2D GpgpuSneCompute<sparse_scalar_matrix_type>::computeEmbeddingBounds(const embedding_type* embedding, float padding) {
      const float* const points = embedding->getContainer().data();

      Bounds2D bounds;
      bounds.min.x = std::numeric_limits<float>::max();
      bounds.max.x = -std::numeric_limits<float>::max();
      bounds.min.y = std::numeric_limits<float>::max();
      bounds.max.y = -std::numeric_limits<float>::max();

      for (int i = 0; i < embedding->numDataPoints(); ++i) {
        float x = points[i * 2 + 0];
        float y = points[i * 2 + 1];

        bounds.min.x = std::min<float>(x, bounds.min.x);
        bounds.max.x = std::max<float>(x, bounds.max.x);
        bounds.min.y = std::min<float>(y, bounds.min.y);
        bounds.max.y = std::max<float>(y, bounds.max.y);
      }

      // Add any extra padding if requested
      if (padding != 0) {
        float half_padding = padding / 2;

        float x_padding = (bounds.max.x - bounds.min.x) * half_padding;
        float y_padding = (bounds.max.y - bounds.min.y) * half_padding;

        bounds.min.x -= x_padding;
        bounds.max.x += x_padding;
        bounds.min.y -= y_padding;
        bounds.max.y += y_padding;
      }

      return bounds;
    }
    template typename GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::Bounds2D GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::computeEmbeddingBounds(const embedding_type*, float);
    template typename GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::Bounds2D GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::computeEmbeddingBounds(const embedding_type*, float);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::initialize(const embedding_type* embedding, TsneParameters params, const sparse_scalar_matrix_type& P) {
      _params = params;

      unsigned int num_points = embedding->numDataPoints();

      // Linearize sparse probability matrix
      LinearProbabilityMatrix linear_P;
      for (int i = 0; i < num_points; ++i) {
        linear_P.indices.push_back(linear_P.neighbours.size());
        int size = 0;

        if constexpr (std::is_same_v<sparse_scalar_matrix_type, std::vector<hdi::data::SparseVec<uint32_t, float>>>)
        {
          for (Eigen::SparseVector<float>::InnerIterator it(P[i].memory()); it; ++it) {
            linear_P.neighbours.push_back(it.index());
            linear_P.probabilities.push_back(it.value());
            size++;
          }
        }
        else // MapMemEff
        {
          for (const auto& pij : P[i]) {
            linear_P.neighbours.push_back(pij.first);
            linear_P.probabilities.push_back(pij.second);
            size++;
          }
        }
        linear_P.indices.push_back(size);
      }

      // Compute initial data bounds
      _bounds = computeEmbeddingBounds(embedding);

      _function_support = 6.5f;

      // Initialize all OpenGL resources
      initializeOpenGL(num_points, linear_P);

      _initialized = true;
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::initialize(const embedding_type*, TsneParameters, const std::vector<hdi::data::SparseVec<uint32_t, float>>&);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::initialize(const embedding_type*, TsneParameters, const std::vector<hdi::data::MapMemEff<uint32_t, float>>&);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::clean()
    {
      glDeleteBuffers(10, _compute_buffers.data());

      fieldComputation.clean();
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::clean();
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::clean();

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::initializeOpenGL(const unsigned int num_points, const LinearProbabilityMatrix& linear_P) {
      glClearColor(0, 0, 0, 0);

      fieldComputation.init(num_points);

      // Load in shader programs
      try {
        _interp_program.create();
        _forces_program.create();
        _update_program.create();
        _bounds_program.create();
        _center_and_scale_program.create();

        _interp_program.addShader(COMPUTE, interp_fields_source);
        _forces_program.addShader(COMPUTE, compute_forces_source);
        _update_program.addShader(COMPUTE, update_source);
        _bounds_program.addShader(COMPUTE, bounds_source);
        _center_and_scale_program.addShader(COMPUTE, center_and_scale_source);

        _interp_program.build();
        _forces_program.build();
        _update_program.build();
        _bounds_program.build();
        _center_and_scale_program.build();
      }
      catch (const ShaderLoadingException& e) {
        std::cout << e.what() << std::endl;
      }

      // Set constant uniforms
      _interp_program.bind();
      _interp_program.uniform1ui("num_points", num_points);
      _forces_program.bind();
      _forces_program.uniform1ui("num_points", num_points);

      // Create the SSBOs
      glGenBuffers(_compute_buffers.size(), _compute_buffers.data());

      // Load up SSBOs with initial values
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), nullptr, GL_STREAM_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[INTERP_FIELDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * 4 * sizeof(float), nullptr, GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[SUM_Q]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), nullptr, GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), nullptr, GL_STREAM_READ);

      // Upload sparse probability matrix
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[NEIGHBOUR]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, linear_P.neighbours.size() * sizeof(uint32_t), linear_P.neighbours.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[PROBABILITIES]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, linear_P.probabilities.size() * sizeof(float), linear_P.probabilities.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[INDEX]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, linear_P.indices.size() * sizeof(int), linear_P.indices.data(), GL_STATIC_DRAW);

      // Initialize buffer with 0s
      std::vector<float> zeroes(num_points * 2, 0);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[PREV_GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), zeroes.data(), GL_STREAM_READ);

      // Initialize buffer with 1s
      std::vector<float> ones(num_points * 2, 1);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[GAIN]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), ones.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[BOUNDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(Point2D), ones.data(), GL_STREAM_READ);

      glGenQueries(2, _timerQuery);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::initializeOpenGL(const unsigned int, const LinearProbabilityMatrix&);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::initializeOpenGL(const unsigned int, const LinearProbabilityMatrix&);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::startTimer()
    {
      glQueryCounter(_timerQuery[0], GL_TIMESTAMP);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::startTimer();
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::startTimer();

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::stopTimer()
    {
      glQueryCounter(_timerQuery[1], GL_TIMESTAMP);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::stopTimer();
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::stopTimer();

    template <typename sparse_scalar_matrix_type>
    double GpgpuSneCompute<sparse_scalar_matrix_type>::getElapsed()
    {
      GLint stopTimerAvailable = 0;
      while (!stopTimerAvailable)
      {
        glGetQueryObjectiv(_timerQuery[1], GL_QUERY_RESULT_AVAILABLE, &stopTimerAvailable);
      }
      GLuint64 startTime, stopTime;
      glGetQueryObjectui64v(_timerQuery[0], GL_QUERY_RESULT, &startTime);
      glGetQueryObjectui64v(_timerQuery[1], GL_QUERY_RESULT, &stopTime);

      double elapsed = (stopTime - startTime) / 1000000.0;

      return elapsed;
    }
    template double GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::getElapsed();
    template double GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::getElapsed();

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::compute(embedding_type* embedding, float exaggeration, float iteration, float mult) {
      float* points = embedding->getContainer().data();
      unsigned int num_points = embedding->numDataPoints();

      if (iteration < 0.5)
      {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), points);
      }

      // Compute the bounds of the given embedding and add a 10% border around it
      computeEmbeddingBounds1(num_points, points, 0.1f);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[BOUNDS]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Point2D) * 2, &_bounds);
      Point2D range = _bounds.getRange();

      float aspect = range.x / range.y;

      uint32_t width = _adaptive_resolution ? std::max((unsigned int)(range.x * _resolutionScaling), MINIMUM_FIELDS_SIZE) : (int)(FIXED_FIELDS_SIZE * aspect);
      uint32_t height = _adaptive_resolution ? std::max((unsigned int)(range.y * _resolutionScaling), MINIMUM_FIELDS_SIZE) : FIXED_FIELDS_SIZE;

      // Compute the fields texture
      fieldComputation.compute(width, height, _function_support, num_points, _compute_buffers[POSITION], _compute_buffers[BOUNDS], _bounds.min.x, _bounds.min.y, _bounds.max.x, _bounds.max.y);

      // Calculate the normalization sum and sample the field values for every point
      float sum_Q = 0;
      interpolateFields(&sum_Q);

      // If normalization sum is 0, cancel further updating
      if (sum_Q == 0) {
        return;
      }

      // Compute the gradients of the KL-function
      computeGradients(num_points, sum_Q, exaggeration);

      // Update the point positions
      updatePoints(num_points, points, embedding, iteration, mult);
      computeEmbeddingBounds1(num_points, points);
      updateEmbedding(num_points, exaggeration, iteration, mult);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, embedding->numDataPoints() * sizeof(Point2D), points);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::compute(embedding_type*, float, float, float);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::compute(embedding_type*, float, float, float);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::computeEmbeddingBounds1(unsigned int num_points, const float* points, float padding, bool square)
    {
      // Compute bounds
      _bounds_program.bind();

      _bounds_program.uniform1ui("num_points", num_points);
      _bounds_program.uniform1f("padding", padding);
      //_bounds_program.uniform1i("square", square);

      // Bind required buffers to shader programGpgpuSneCompute
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[BOUNDS]);

      // Compute the bounds
      glDispatchCompute(1, 1, 1);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::computeEmbeddingBounds1(unsigned int, const float*, float, bool);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::computeEmbeddingBounds1(unsigned int, const float*, float, bool);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::interpolateFields(float* sum_Q)
    {
      // Bind fields texture for bilinear sampling
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, fieldComputation.getFieldTexture());

      // Set shader and uniforms
      _interp_program.bind();
      _interp_program.uniform1i("fields", 0);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[INTERP_FIELDS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[SUM_Q]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _compute_buffers[BOUNDS]);

      // Sample the fields texture for every point
      glDispatchCompute(1, 1, 1);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      // Copy sum_Q back to CPU
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[SUM_Q]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), sum_Q);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::interpolateFields(float*);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::interpolateFields(float*);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::computeGradients(unsigned int num_points, float sum_Q, double exaggeration)
    {
      _forces_program.bind();

      _forces_program.uniform1f("exaggeration", exaggeration);
      _forces_program.uniform1f("sum_Q", sum_Q);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[NEIGHBOUR]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[PROBABILITIES]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _compute_buffers[INDEX]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _compute_buffers[INTERP_FIELDS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _compute_buffers[GRADIENTS]);

      // Compute the gradients of the KL function
      unsigned int grid_size = static_cast<unsigned int>(std::sqrt(num_points)) + 1;
      glDispatchCompute(grid_size, grid_size, 1);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::computeGradients(unsigned int, float, double);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::computeGradients(unsigned int, float, double);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::updatePoints(unsigned int num_points, float* points, embedding_type* embedding, float iteration, float mult)
    {
      _update_program.bind();

      _update_program.uniform1ui("num_points", num_points);
      _update_program.uniform1f("eta", _params._eta);
      _update_program.uniform1f("minGain", _params._minimum_gain);
      _update_program.uniform1f("iter", iteration);
      _update_program.uniform1f("mom_iter", _params._mom_switching_iter);
      _update_program.uniform1f("mom", _params._momentum);
      _update_program.uniform1f("final_mom", _params._final_momentum);
      _update_program.uniform1f("mult", mult);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[GRADIENTS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[PREV_GRADIENTS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _compute_buffers[GAIN]);

      // Update the points
      unsigned int num_workgroups = (num_points * 2 / 64) + 1;
      unsigned int grid_size = static_cast<unsigned int>(std::sqrt(num_workgroups)) + 1;
      glDispatchCompute(grid_size, grid_size, 1);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::updatePoints(unsigned int, float*, embedding_type*, float, float);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::updatePoints(unsigned int, float*, embedding_type*, float, float);

    template <typename sparse_scalar_matrix_type>
    void GpgpuSneCompute<sparse_scalar_matrix_type>::updateEmbedding(unsigned int num_points, float exaggeration, float iteration, float mult) {
      _center_and_scale_program.bind();

      _center_and_scale_program.uniform1ui("num_points", num_points);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[BOUNDS]);

      if (exaggeration > 1.2)
      {
        _center_and_scale_program.uniform1i("scale", 1);
        _center_and_scale_program.uniform1f("diameter", 0.1f);
      }
      else
      {
        _center_and_scale_program.uniform1i("scale", 0);
      }

      // Compute the bounds
      unsigned int num_workgroups = (num_points / 128) + 1;
      unsigned int grid_size = static_cast<unsigned int>(std::sqrt(num_workgroups)) + 1;
      glDispatchCompute(grid_size, grid_size, 1);
    }
    template void GpgpuSneCompute<std::vector<hdi::data::SparseVec<uint32_t, float>>>::updateEmbedding(unsigned int, float, float, float);
    template void GpgpuSneCompute<std::vector<hdi::data::MapMemEff<uint32_t, float>>>::updateEmbedding(unsigned int, float, float, float);

  }
}


#endif // __APPLE__
